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
    struct tuple1
    {
        typedef A0 member_type0; A0 a0;
        template <int E>
        typename detail::tuple_element<E, tuple1>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple1>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple1 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple1 const>::get(*this);
        }
        tuple1() {}
        tuple1(tuple1 const& other)
          : a0(other.a0)
        {}
        tuple1(BOOST_RV_REF(tuple1) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0))
        {}
        tuple1 & operator=(BOOST_COPY_ASSIGN_REF(tuple1) other)
        {
            a0 = other.a0;
            return *this;
        }
        tuple1 & operator=(BOOST_RV_REF(tuple1) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0);
            return *this;
        }
        template <typename T0>
        tuple1 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple1<T0>
            ))) other)
        {
            a0 = other.a0;
            return *this;
        }
        template <typename T0>
        tuple1 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple1<T0>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0);
            return *this;
        }
        void swap(tuple1& other)
        {
            boost::swap(a0, other.a0);
        }
        template <typename Arg0>
        explicit tuple1(BOOST_FWD_REF(Arg0) arg0,
                typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : a0(boost::forward<Arg0>(arg0))
        {}
        struct forwarding_tag {};
        template <typename Arg0>
        explicit tuple1(BOOST_FWD_REF(Arg0) arg0, forwarding_tag)
          : a0(boost::forward<Arg0>(arg0))
        {}
        template <typename Arg0>
        explicit tuple1(BOOST_RV_REF(tuple<Arg0>) arg0)
          : a0(detail::move_if_no_ref<Arg0>::call(arg0.a0))
        {}
        template <typename Arg0>
        explicit tuple1(BOOST_RV_REF(tuple1<Arg0>) arg0)
          : a0(detail::move_if_no_ref<Arg0>::call(arg0.a0))
        {}
        template <typename Arg0>
        explicit tuple1(tuple<Arg0> const& arg0)
          : a0(arg0.a0)
        {}
        template <typename Arg0>
        explicit tuple1(tuple1<Arg0> const& arg0)
          : a0(arg0.a0)
        {}
        typedef boost::mpl::int_<1> size_type;
        static const int size_value = 1;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple1);
    };
    template <typename T0>
    struct is_tuple<tuple1<T0> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple1<Arg0>
    tie(Arg0& arg0) BOOST_NOEXCEPT
    {
        typedef tuple1<Arg0> result_type;
        return result_type(arg0,
            typename result_type::forwarding_tag());
    }
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple1<typename detail::env_value_type<Arg0>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0) BOOST_NOEXCEPT
    {
        typedef tuple1<typename detail::env_value_type<Arg0>::type> result_type;
        return result_type(boost::forward<Arg0>(arg0),
            typename result_type::forwarding_tag());
    }
    
    template <typename A0>
    struct tuple<A0>
      : tuple1<A0>
    {
        typedef tuple1<A0> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple1<T0>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0>
        explicit tuple(BOOST_FWD_REF(Arg0) arg0,
                typename boost::disable_if<is_tuple<Arg0> >::type* = 0)
          : base_tuple(boost::forward<Arg0>(arg0))
        {}
        struct forwarding_tag {};
        template <typename Arg0>
        explicit tuple(BOOST_FWD_REF(Arg0) arg0, forwarding_tag t)
          : base_tuple(boost::forward<Arg0>(arg0), typename base_tuple::forwarding_tag())
        {}
        template <typename Arg0>
        explicit tuple(BOOST_RV_REF(tuple<Arg0>) arg0)
          : base_tuple(boost::move(arg0))
        {}
        template <typename Arg0>
        explicit tuple(BOOST_RV_REF(tuple1<Arg0>) arg0)
          : base_tuple(boost::move(arg0))
        {}
        template <typename Arg0>
        explicit tuple(tuple<Arg0> const& arg0)
          : base_tuple(arg0)
        {}
        template <typename Arg0>
        explicit tuple(tuple1<Arg0> const& arg0)
          : base_tuple(arg0)
        {}
    };
    
    template <typename Arg0>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0)
    {
        typedef tuple<typename util::decay<Arg0>::type> result_type;
        return result_type(boost::forward<Arg0>(arg0),
            typename result_type::forwarding_tag());
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0 > struct tag_of<hpx::util::tuple1<A0> > { typedef struct_tag type; }; template< typename A0 > struct tag_of<hpx::util::tuple1<A0> const> { typedef struct_tag type; }; } namespace extension { template< typename A0 > struct access::struct_member< hpx::util::tuple1<A0> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0 > struct struct_member_name< hpx::util::tuple1<A0> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0 > struct struct_size<hpx::util::tuple1<A0> > : mpl::int_<1> {}; template< typename A0 > struct struct_is_view< hpx::util::tuple1<A0> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0 > struct sequence_tag<hpx::util::tuple1<A0> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0 > struct sequence_tag< hpx::util::tuple1<A0> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0 > struct tag_of<hpx::util::tuple<A0> > { typedef struct_tag type; }; template< typename A0 > struct tag_of<hpx::util::tuple<A0> const> { typedef struct_tag type; }; } namespace extension { template< typename A0 > struct access::struct_member< hpx::util::tuple<A0> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0 > struct struct_member_name< hpx::util::tuple<A0> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0 > struct struct_size<hpx::util::tuple<A0> > : mpl::int_<1> {}; template< typename A0 > struct struct_is_view< hpx::util::tuple<A0> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0 > struct sequence_tag<hpx::util::tuple<A0> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0 > struct sequence_tag< hpx::util::tuple<A0> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0>
    struct is_bitwise_serializable<
            hpx::util::tuple1<T0> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple1<T0> >
    {};
    template <typename T0>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0> >
    {};
    
    template <typename Archive, typename T0>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple1<T0>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple2
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1;
        template <int E>
        typename detail::tuple_element<E, tuple2>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple2>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple2 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple2 const>::get(*this);
        }
        tuple2() {}
        tuple2(tuple2 const& other)
          : a0(other.a0) , a1(other.a1)
        {}
        tuple2(BOOST_RV_REF(tuple2) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1))
        {}
        tuple2 & operator=(BOOST_COPY_ASSIGN_REF(tuple2) other)
        {
            a0 = other.a0; a1 = other.a1;
            return *this;
        }
        tuple2 & operator=(BOOST_RV_REF(tuple2) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1);
            return *this;
        }
        template <typename T0 , typename T1>
        tuple2 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple2<T0 , T1>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1;
            return *this;
        }
        template <typename T0 , typename T1>
        tuple2 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple2<T0 , T1>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1);
            return *this;
        }
        void swap(tuple2& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1);
        }
        template <typename Arg0 , typename Arg1>
        tuple2(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 ))
        {}
        template <typename T0 , typename T1>
        tuple2(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple2<T0 , T1>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1))
        {}
        template <typename T0 , typename T1>
        tuple2(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1))
        {}
        typedef boost::mpl::int_<2> size_type;
        static const int size_value = 2;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple2);
    };
    template <typename T0 , typename T1>
    struct is_tuple<tuple2<T0 , T1> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple2<Arg0 & , Arg1 &>
    tie(Arg0 & arg0 , Arg1 & arg1) BOOST_NOEXCEPT
    {
        return tuple2<
                Arg0 & , Arg1 &>(
            arg0 , arg1);
    }
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple2<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1) BOOST_NOEXCEPT
    {
        return tuple2<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
    
    template <typename A0 , typename A1>
    struct tuple<A0 , A1>
      : tuple2<A0 , A1>
    {
        typedef tuple2<A0 , A1> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple2<T0 , T1>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ))
        {}
        template <typename T0 , typename T1>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple2<T0 , T1>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1 > struct tag_of<hpx::util::tuple2<A0, A1> > { typedef struct_tag type; }; template< typename A0, typename A1 > struct tag_of<hpx::util::tuple2<A0, A1> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple2<A0, A1> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple2<A0, A1> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple2<A0, A1> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple2<A0, A1> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1 > struct struct_size<hpx::util::tuple2<A0, A1> > : mpl::int_<2> {}; template< typename A0, typename A1 > struct struct_is_view< hpx::util::tuple2<A0, A1> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1 > struct sequence_tag<hpx::util::tuple2<A0, A1> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1 > struct sequence_tag< hpx::util::tuple2<A0, A1> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1 > struct tag_of<hpx::util::tuple<A0, A1> > { typedef struct_tag type; }; template< typename A0, typename A1 > struct tag_of<hpx::util::tuple<A0, A1> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple<A0, A1> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple<A0, A1> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1 > struct access::struct_member< hpx::util::tuple<A0, A1> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1 > struct struct_member_name< hpx::util::tuple<A0, A1> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1 > struct struct_size<hpx::util::tuple<A0, A1> > : mpl::int_<2> {}; template< typename A0, typename A1 > struct struct_is_view< hpx::util::tuple<A0, A1> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1 > struct sequence_tag<hpx::util::tuple<A0, A1> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1 > struct sequence_tag< hpx::util::tuple<A0, A1> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1>
    struct is_bitwise_serializable<
            hpx::util::tuple2<T0 , T1> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple2<T0 , T1> >
    {};
    template <typename T0 , typename T1>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1> >
    {};
    
    template <typename Archive, typename T0 , typename T1>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple2<T0 , T1>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple3
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2;
        template <int E>
        typename detail::tuple_element<E, tuple3>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple3>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple3 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple3 const>::get(*this);
        }
        tuple3() {}
        tuple3(tuple3 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2)
        {}
        tuple3(BOOST_RV_REF(tuple3) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2))
        {}
        tuple3 & operator=(BOOST_COPY_ASSIGN_REF(tuple3) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2;
            return *this;
        }
        tuple3 & operator=(BOOST_RV_REF(tuple3) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple3 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple3<T0 , T1 , T2>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple3 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple3<T0 , T1 , T2>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2);
            return *this;
        }
        void swap(tuple3& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2>
        tuple3(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 ))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple3(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple3<T0 , T1 , T2>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple3(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2))
        {}
        typedef boost::mpl::int_<3> size_type;
        static const int size_value = 3;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple3);
    };
    template <typename T0 , typename T1 , typename T2>
    struct is_tuple<tuple3<T0 , T1 , T2> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple3<Arg0 & , Arg1 & , Arg2 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2) BOOST_NOEXCEPT
    {
        return tuple3<
                Arg0 & , Arg1 & , Arg2 &>(
            arg0 , arg1 , arg2);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple3<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2) BOOST_NOEXCEPT
    {
        return tuple3<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
    
    template <typename A0 , typename A1 , typename A2>
    struct tuple<A0 , A1 , A2>
      : tuple3<A0 , A1 , A2>
    {
        typedef tuple3<A0 , A1 , A2> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple3<T0 , T1 , T2>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ))
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple3<T0 , T1 , T2>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple3<A0, A1, A2> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple3<A0, A1, A2> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple3<A0, A1, A2> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple3<A0, A1, A2> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple3<A0, A1, A2> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple3<A0, A1, A2> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple3<A0, A1, A2> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple3<A0, A1, A2> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2 > struct struct_size<hpx::util::tuple3<A0, A1, A2> > : mpl::int_<3> {}; template< typename A0, typename A1, typename A2 > struct struct_is_view< hpx::util::tuple3<A0, A1, A2> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2 > struct sequence_tag<hpx::util::tuple3<A0, A1, A2> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2 > struct sequence_tag< hpx::util::tuple3<A0, A1, A2> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple<A0, A1, A2> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2 > struct tag_of<hpx::util::tuple<A0, A1, A2> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2 > struct access::struct_member< hpx::util::tuple<A0, A1, A2> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2 > struct struct_member_name< hpx::util::tuple<A0, A1, A2> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2 > struct struct_size<hpx::util::tuple<A0, A1, A2> > : mpl::int_<3> {}; template< typename A0, typename A1, typename A2 > struct struct_is_view< hpx::util::tuple<A0, A1, A2> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2 > struct sequence_tag<hpx::util::tuple<A0, A1, A2> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2 > struct sequence_tag< hpx::util::tuple<A0, A1, A2> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2>
    struct is_bitwise_serializable<
            hpx::util::tuple3<T0 , T1 , T2> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple3<T0 , T1 , T2> >
    {};
    template <typename T0 , typename T1 , typename T2>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple3<T0 , T1 , T2>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple4
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3;
        template <int E>
        typename detail::tuple_element<E, tuple4>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple4>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple4 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple4 const>::get(*this);
        }
        tuple4() {}
        tuple4(tuple4 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3)
        {}
        tuple4(BOOST_RV_REF(tuple4) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3))
        {}
        tuple4 & operator=(BOOST_COPY_ASSIGN_REF(tuple4) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3;
            return *this;
        }
        tuple4 & operator=(BOOST_RV_REF(tuple4) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple4 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple4<T0 , T1 , T2 , T3>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple4 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple4<T0 , T1 , T2 , T3>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3);
            return *this;
        }
        void swap(tuple4& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        tuple4(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple4(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple4<T0 , T1 , T2 , T3>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple4(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3))
        {}
        typedef boost::mpl::int_<4> size_type;
        static const int size_value = 4;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple4);
    };
    template <typename T0 , typename T1 , typename T2 , typename T3>
    struct is_tuple<tuple4<T0 , T1 , T2 , T3> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple4<Arg0 & , Arg1 & , Arg2 & , Arg3 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3) BOOST_NOEXCEPT
    {
        return tuple4<
                Arg0 & , Arg1 & , Arg2 & , Arg3 &>(
            arg0 , arg1 , arg2 , arg3);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple4<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3) BOOST_NOEXCEPT
    {
        return tuple4<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
    
    template <typename A0 , typename A1 , typename A2 , typename A3>
    struct tuple<A0 , A1 , A2 , A3>
      : tuple4<A0 , A1 , A2 , A3>
    {
        typedef tuple4<A0 , A1 , A2 , A3> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple4<T0 , T1 , T2 , T3>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple4<T0 , T1 , T2 , T3>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple4<A0, A1, A2, A3> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple4<A0, A1, A2, A3> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple4<A0, A1, A2, A3> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple4<A0, A1, A2, A3> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple4<A0, A1, A2, A3> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple4<A0, A1, A2, A3> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple4<A0, A1, A2, A3> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple4<A0, A1, A2, A3> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple4<A0, A1, A2, A3> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple4<A0, A1, A2, A3> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_size<hpx::util::tuple4<A0, A1, A2, A3> > : mpl::int_<4> {}; template< typename A0, typename A1, typename A2, typename A3 > struct struct_is_view< hpx::util::tuple4<A0, A1, A2, A3> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag<hpx::util::tuple4<A0, A1, A2, A3> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag< hpx::util::tuple4<A0, A1, A2, A3> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3> > : mpl::int_<4> {}; template< typename A0, typename A1, typename A2, typename A3 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    struct is_bitwise_serializable<
            hpx::util::tuple4<T0 , T1 , T2 , T3> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple4<T0 , T1 , T2 , T3> >
    {};
    template <typename T0 , typename T1 , typename T2 , typename T3>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple4<T0 , T1 , T2 , T3>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple5
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4;
        template <int E>
        typename detail::tuple_element<E, tuple5>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple5>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple5 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple5 const>::get(*this);
        }
        tuple5() {}
        tuple5(tuple5 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4)
        {}
        tuple5(BOOST_RV_REF(tuple5) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4))
        {}
        tuple5 & operator=(BOOST_COPY_ASSIGN_REF(tuple5) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4;
            return *this;
        }
        tuple5 & operator=(BOOST_RV_REF(tuple5) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple5 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple5<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple5 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple5<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4);
            return *this;
        }
        void swap(tuple5& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        tuple5(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple5(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple5<T0 , T1 , T2 , T3 , T4>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple5(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4))
        {}
        typedef boost::mpl::int_<5> size_type;
        static const int size_value = 5;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple5);
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct is_tuple<tuple5<T0 , T1 , T2 , T3 , T4> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple5<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4) BOOST_NOEXCEPT
    {
        return tuple5<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 &>(
            arg0 , arg1 , arg2 , arg3 , arg4);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple5<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4) BOOST_NOEXCEPT
    {
        return tuple5<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    struct tuple<A0 , A1 , A2 , A3 , A4>
      : tuple5<A0 , A1 , A2 , A3 , A4>
    {
        typedef tuple5<A0 , A1 , A2 , A3 , A4> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple5<T0 , T1 , T2 , T3 , T4>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple5<T0 , T1 , T2 , T3 , T4>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple5<A0, A1, A2, A3, A4> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple5<A0, A1, A2, A3, A4> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple5<A0, A1, A2, A3, A4> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple5<A0, A1, A2, A3, A4> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple5<A0, A1, A2, A3, A4> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple5<A0, A1, A2, A3, A4> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple5<A0, A1, A2, A3, A4> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple5<A0, A1, A2, A3, A4> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple5<A0, A1, A2, A3, A4> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple5<A0, A1, A2, A3, A4> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple5<A0, A1, A2, A3, A4> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple5<A0, A1, A2, A3, A4> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_size<hpx::util::tuple5<A0, A1, A2, A3, A4> > : mpl::int_<5> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_is_view< hpx::util::tuple5<A0, A1, A2, A3, A4> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag<hpx::util::tuple5<A0, A1, A2, A3, A4> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag< hpx::util::tuple5<A0, A1, A2, A3, A4> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4> > : mpl::int_<5> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct is_bitwise_serializable<
            hpx::util::tuple5<T0 , T1 , T2 , T3 , T4> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple5<T0 , T1 , T2 , T3 , T4> >
    {};
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple5<T0 , T1 , T2 , T3 , T4>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple6
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5;
        template <int E>
        typename detail::tuple_element<E, tuple6>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple6>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple6 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple6 const>::get(*this);
        }
        tuple6() {}
        tuple6(tuple6 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5)
        {}
        tuple6(BOOST_RV_REF(tuple6) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5))
        {}
        tuple6 & operator=(BOOST_COPY_ASSIGN_REF(tuple6) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5;
            return *this;
        }
        tuple6 & operator=(BOOST_RV_REF(tuple6) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple6 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple6<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple6 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple6<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5);
            return *this;
        }
        void swap(tuple6& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        tuple6(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple6(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple6<T0 , T1 , T2 , T3 , T4 , T5>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple6(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5))
        {}
        typedef boost::mpl::int_<6> size_type;
        static const int size_value = 6;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple6);
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct is_tuple<tuple6<T0 , T1 , T2 , T3 , T4 , T5> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple6<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5) BOOST_NOEXCEPT
    {
        return tuple6<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple6<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5) BOOST_NOEXCEPT
    {
        return tuple6<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5>
      : tuple6<A0 , A1 , A2 , A3 , A4 , A5>
    {
        typedef tuple6<A0 , A1 , A2 , A3 , A4 , A5> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple6<T0 , T1 , T2 , T3 , T4 , T5>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple6<T0 , T1 , T2 , T3 , T4 , T5>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple6<A0, A1, A2, A3, A4, A5> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple6<A0, A1, A2, A3, A4, A5> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_size<hpx::util::tuple6<A0, A1, A2, A3, A4, A5> > : mpl::int_<6> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_is_view< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag<hpx::util::tuple6<A0, A1, A2, A3, A4, A5> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag< hpx::util::tuple6<A0, A1, A2, A3, A4, A5> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > : mpl::int_<6> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct is_bitwise_serializable<
            hpx::util::tuple6<T0 , T1 , T2 , T3 , T4 , T5> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple6<T0 , T1 , T2 , T3 , T4 , T5> >
    {};
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple6<T0 , T1 , T2 , T3 , T4 , T5>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple7
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6;
        template <int E>
        typename detail::tuple_element<E, tuple7>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple7>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple7 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple7 const>::get(*this);
        }
        tuple7() {}
        tuple7(tuple7 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6)
        {}
        tuple7(BOOST_RV_REF(tuple7) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5)) , a6(detail::move_if_no_ref<A6>::call( other.a6))
        {}
        tuple7 & operator=(BOOST_COPY_ASSIGN_REF(tuple7) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6;
            return *this;
        }
        tuple7 & operator=(BOOST_RV_REF(tuple7) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5); a6 = detail::move_if_no_ref<A6>::call( other.a6);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple7 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple7 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5); a6 = detail::move_if_no_ref<T6>::call( other.a6);
            return *this;
        }
        void swap(tuple7& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        tuple7(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple7(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple7(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6))
        {}
        typedef boost::mpl::int_<7> size_type;
        static const int size_value = 7;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple7);
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct is_tuple<tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple7<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6) BOOST_NOEXCEPT
    {
        return tuple7<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple7<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6) BOOST_NOEXCEPT
    {
        return tuple7<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6>
      : tuple7<A0 , A1 , A2 , A3 , A4 , A5 , A6>
    {
        typedef tuple7<A0 , A1 , A2 , A3 , A4 , A5 , A6> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_size<hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> > : mpl::int_<7> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_is_view< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag<hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag< hpx::util::tuple7<A0, A1, A2, A3, A4, A5, A6> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > : mpl::int_<7> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct is_bitwise_serializable<
            hpx::util::tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
    {};
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple7<T0 , T1 , T2 , T3 , T4 , T5 , T6>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
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
    struct tuple8
    {
        typedef A0 member_type0; A0 a0; typedef A1 member_type1; A1 a1; typedef A2 member_type2; A2 a2; typedef A3 member_type3; A3 a3; typedef A4 member_type4; A4 a4; typedef A5 member_type5; A5 a5; typedef A6 member_type6; A6 a6; typedef A7 member_type7; A7 a7;
        template <int E>
        typename detail::tuple_element<E, tuple8>::rtype
        get() BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple8>::get(*this);
        }
        template <int E>
        BOOST_CONSTEXPR
        typename detail::tuple_element<E, tuple8 const>::crtype
        get() const BOOST_NOEXCEPT
        {
            return detail::tuple_element<E, tuple8 const>::get(*this);
        }
        tuple8() {}
        tuple8(tuple8 const& other)
          : a0(other.a0) , a1(other.a1) , a2(other.a2) , a3(other.a3) , a4(other.a4) , a5(other.a5) , a6(other.a6) , a7(other.a7)
        {}
        tuple8(BOOST_RV_REF(tuple8) other)
          : a0(detail::move_if_no_ref<A0>::call( other.a0)) , a1(detail::move_if_no_ref<A1>::call( other.a1)) , a2(detail::move_if_no_ref<A2>::call( other.a2)) , a3(detail::move_if_no_ref<A3>::call( other.a3)) , a4(detail::move_if_no_ref<A4>::call( other.a4)) , a5(detail::move_if_no_ref<A5>::call( other.a5)) , a6(detail::move_if_no_ref<A6>::call( other.a6)) , a7(detail::move_if_no_ref<A7>::call( other.a7))
        {}
        tuple8 & operator=(BOOST_COPY_ASSIGN_REF(tuple8) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7;
            return *this;
        }
        tuple8 & operator=(BOOST_RV_REF(tuple8) other)
        {
            a0 = detail::move_if_no_ref<A0>::call( other.a0); a1 = detail::move_if_no_ref<A1>::call( other.a1); a2 = detail::move_if_no_ref<A2>::call( other.a2); a3 = detail::move_if_no_ref<A3>::call( other.a3); a4 = detail::move_if_no_ref<A4>::call( other.a4); a5 = detail::move_if_no_ref<A5>::call( other.a5); a6 = detail::move_if_no_ref<A6>::call( other.a6); a7 = detail::move_if_no_ref<A7>::call( other.a7);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple8 & operator=(BOOST_COPY_ASSIGN_REF(HPX_UTIL_STRIP((
                tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            a0 = other.a0; a1 = other.a1; a2 = other.a2; a3 = other.a3; a4 = other.a4; a5 = other.a5; a6 = other.a6; a7 = other.a7;
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple8 & operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            a0 = detail::move_if_no_ref<T0>::call( other.a0); a1 = detail::move_if_no_ref<T1>::call( other.a1); a2 = detail::move_if_no_ref<T2>::call( other.a2); a3 = detail::move_if_no_ref<T3>::call( other.a3); a4 = detail::move_if_no_ref<T4>::call( other.a4); a5 = detail::move_if_no_ref<T5>::call( other.a5); a6 = detail::move_if_no_ref<T6>::call( other.a6); a7 = detail::move_if_no_ref<T7>::call( other.a7);
            return *this;
        }
        void swap(tuple8& other)
        {
            boost::swap(a0, other.a0); boost::swap(a1, other.a1); boost::swap(a2, other.a2); boost::swap(a3, other.a3); boost::swap(a4, other.a4); boost::swap(a5, other.a5); boost::swap(a6, other.a6); boost::swap(a7, other.a7);
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        tuple8(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
          : a0(boost::forward<Arg0>( arg0 )) , a1(boost::forward<Arg1>( arg1 )) , a2(boost::forward<Arg2>( arg2 )) , a3(boost::forward<Arg3>( arg3 )) , a4(boost::forward<Arg4>( arg4 )) , a5(boost::forward<Arg5>( arg5 )) , a6(boost::forward<Arg6>( arg6 )) , a7(boost::forward<Arg7>( arg7 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple8(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6)) , a7(detail::move_if_no_ref<T7>::call( other.a7))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple8(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
                ))) other)
          : a0(detail::move_if_no_ref<T0>::call( other.a0)) , a1(detail::move_if_no_ref<T1>::call( other.a1)) , a2(detail::move_if_no_ref<T2>::call( other.a2)) , a3(detail::move_if_no_ref<T3>::call( other.a3)) , a4(detail::move_if_no_ref<T4>::call( other.a4)) , a5(detail::move_if_no_ref<T5>::call( other.a5)) , a6(detail::move_if_no_ref<T6>::call( other.a6)) , a7(detail::move_if_no_ref<T7>::call( other.a7))
        {}
        typedef boost::mpl::int_<8> size_type;
        static const int size_value = 8;
    private:
        BOOST_COPYABLE_AND_MOVABLE(tuple8);
    };
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct is_tuple<tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
      : boost::mpl::true_
    {};
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple8<Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 &>
    tie(Arg0 & arg0 , Arg1 & arg1 , Arg2 & arg2 , Arg3 & arg3 , Arg4 & arg4 , Arg5 & arg5 , Arg6 & arg6 , Arg7 & arg7) BOOST_NOEXCEPT
    {
        return tuple8<
                Arg0 & , Arg1 & , Arg2 & , Arg3 & , Arg4 & , Arg5 & , Arg6 & , Arg7 &>(
            arg0 , arg1 , arg2 , arg3 , arg4 , arg5 , arg6 , arg7);
    }
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple8<typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type , typename detail::env_value_type<Arg7>::type>
    forward_as_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7) BOOST_NOEXCEPT
    {
        return tuple8<
                typename detail::env_value_type<Arg0>::type , typename detail::env_value_type<Arg1>::type , typename detail::env_value_type<Arg2>::type , typename detail::env_value_type<Arg3>::type , typename detail::env_value_type<Arg4>::type , typename detail::env_value_type<Arg5>::type , typename detail::env_value_type<Arg6>::type , typename detail::env_value_type<Arg7>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
    
    template <typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    struct tuple<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
      : tuple8<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7>
    {
        typedef tuple8<A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7> base_tuple;
        tuple() {}
        tuple(tuple const& other)
          : base_tuple(other)
        {}
        tuple(base_tuple const& other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(tuple) other)
          : base_tuple(other)
        {}
        tuple(BOOST_RV_REF(base_tuple) other)
          : base_tuple(other)
        {}
        tuple& operator=(BOOST_COPY_ASSIGN_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_COPY_ASSIGN_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        tuple& operator=(BOOST_RV_REF(base_tuple) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple& operator=(BOOST_RV_REF(HPX_UTIL_STRIP((
                tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
            ))) other)
        {
            this->base_tuple::operator=(other);
            return *this;
        }
        template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
        tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
          : base_tuple(boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ))
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
                ))) other)
          : base_tuple(other)
        {}
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        tuple(BOOST_RV_REF(HPX_UTIL_STRIP((
                    tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
                ))) other)
          : base_tuple(other)
        {}
    };
    
    template <typename Arg0 , typename Arg1 , typename Arg2 , typename Arg3 , typename Arg4 , typename Arg5 , typename Arg6 , typename Arg7>
    BOOST_FORCEINLINE
    tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type>
    make_tuple(BOOST_FWD_REF(Arg0) arg0 , BOOST_FWD_REF(Arg1) arg1 , BOOST_FWD_REF(Arg2) arg2 , BOOST_FWD_REF(Arg3) arg3 , BOOST_FWD_REF(Arg4) arg4 , BOOST_FWD_REF(Arg5) arg5 , BOOST_FWD_REF(Arg6) arg6 , BOOST_FWD_REF(Arg7) arg7)
    {
        return tuple<typename util::decay<Arg0>::type , typename util::decay<Arg1>::type , typename util::decay<Arg2>::type , typename util::decay<Arg3>::type , typename util::decay<Arg4>::type , typename util::decay<Arg5>::type , typename util::decay<Arg6>::type , typename util::decay<Arg7>::type>(
            boost::forward<Arg0>( arg0 ) , boost::forward<Arg1>( arg1 ) , boost::forward<Arg2>( arg2 ) , boost::forward<Arg3>( arg3 ) , boost::forward<Arg4>( arg4 ) , boost::forward<Arg5>( arg5 ) , boost::forward<Arg6>( arg6 ) , boost::forward<Arg7>( arg7 ));
    }
}}
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_size<hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::int_<8> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_is_view< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag<hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag< hpx::util::tuple8<A0, A1, A2, A3, A4, A5, A6, A7> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace fusion { namespace traits { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef struct_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct tag_of<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> const> { typedef struct_tag type; }; } namespace extension { template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef A0 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a0; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 0 > { typedef char const* type; static type call() { return "a0"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef A1 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a1; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 1 > { typedef char const* type; static type call() { return "a1"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef A2 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a2; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 2 > { typedef char const* type; static type call() { return "a2"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef A3 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a3; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 3 > { typedef char const* type; static type call() { return "a3"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef A4 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a4; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 4 > { typedef char const* type; static type call() { return "a4"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef A5 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a5; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 5 > { typedef char const* type; static type call() { return "a5"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef A6 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a6; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 6 > { typedef char const* type; static type call() { return "a6"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct access::struct_member< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef A7 attribute_type; BOOST_FUSION_ADAPT_STRUCT_MSVC_REDEFINE_TEMPLATE_PARAMS( (1) (A0) (A1) (A2) (A3) (A4) (A5) (A6) (A7)) typedef attribute_type type; template<typename Seq> struct apply { typedef typename add_reference< typename mpl::eval_if< is_const<Seq> , add_const<attribute_type> , mpl::identity<attribute_type> >::type >::type type; static type call(Seq& seq) { return seq. a7; } }; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_member_name< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> , 7 > { typedef char const* type; static type call() { return "a7"; } }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_size<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::int_<8> {}; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct struct_is_view< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > : mpl::false_ {}; } } namespace mpl { template<typename> struct sequence_tag; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag<hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> > { typedef fusion::fusion_sequence_tag type; }; template< typename A0, typename A1, typename A2, typename A3, typename A4, typename A5, typename A6, typename A7 > struct sequence_tag< hpx::util::tuple<A0, A1, A2, A3, A4, A5, A6, A7> const > { typedef fusion::fusion_sequence_tag type; }; } }
namespace boost { namespace serialization
{
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct is_bitwise_serializable<
            hpx::util::tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
       : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
    {};
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
      : hpx::util::detail::sequence_is_bitwise_serializable<
            hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
    {};
    
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple8<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
    template <typename Archive, typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_FORCEINLINE void serialize(Archive& ar,
        hpx::util::tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>& t,
        unsigned int const version)
    {
        hpx::util::serialize_sequence(ar, t);
    }
}}
