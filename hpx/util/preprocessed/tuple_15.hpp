// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0> >::value == 1
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 1
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0>
    class tuple<T0>
    {
    public: 
        detail::tuple_member<T0> _m0;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0
        ) : _m0(v0)
        {}
        
        
        
        
        
        
        
        template <typename U0>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0>&&
                >::value
             && !boost::is_base_of<
                    tuple, typename boost::remove_reference<U0>::type
                 >::value
             && !detail::are_tuples_compatible<tuple, U0&&>::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value ))
            )
        {
            _m0._value = other._m0._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 1
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value );;
        }
    };
    
    
    
    template <typename T0>
    struct tuple_size<tuple<T0> >
      : boost::mpl::size_t<1>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        0
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T0>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T0
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m0._value;
        }
    };
    
    
    
    template <typename T0>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type>
    make_tuple(T0 && v0)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type>(
                std::forward<T0>( v0 )
            );
    }
    
    
    
    
    
    
    template <typename T0>
    BOOST_FORCEINLINE
    tuple<T0 &&>
    forward_as_tuple(T0 && v0) BOOST_NOEXCEPT
    {
        return
            tuple<T0 &&>(
                std::forward<T0>( v0 )
            );
    }
    
    
    template <typename T0>
    BOOST_FORCEINLINE
    tuple<T0 &>
    tie(T0 & v0) BOOST_NOEXCEPT
    {
        return
            tuple<T0 &>(
                v0
            );
    }
    
    
    namespace detail
    {
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 1>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 1
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 1
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1> >::value == 2
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 2
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1>
    class tuple<T0 , T1>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1
        ) : _m0(v0) , _m1(v1)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 2
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1>
    struct tuple_size<tuple<T0 , T1> >
      : boost::mpl::size_t<2>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        1
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T1>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T1
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m1._value;
        }
    };
    
    
    
    template <typename T0 , typename T1>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type>
    make_tuple(T0 && v0 , T1 && v1)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 &&>
    forward_as_tuple(T0 && v0 , T1 && v1) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 )
            );
    }
    
    
    template <typename T0 , typename T1>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 &>
    tie(T0 & v0 , T1 & v1) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 &>(
                v0 , v1
            );
    }
    
    
    namespace detail
    {
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 2>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 2
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 2
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2> >::value == 3
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 3
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2>
    class tuple<T0 , T1 , T2>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2
        ) : _m0(v0) , _m1(v1) , _m2(v2)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 3
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2>
    struct tuple_size<tuple<T0 , T1 , T2> >
      : boost::mpl::size_t<3>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        2
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T2>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T2
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m2._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 &>(
                v0 , v1 , v2
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2>
        struct tuple_cat_result<T0 , T1 , T2>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type
              , T2
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 3>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 3
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 3
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1))
              , std::forward<T2>
                    (t2)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3> >::value == 4
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 4
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    class tuple<T0 , T1 , T2 , T3>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 4
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    struct tuple_size<tuple<T0 , T1 , T2 , T3> >
      : boost::mpl::size_t<4>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        3
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T3>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T3
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m3._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 &>(
                v0 , v1 , v2 , v3
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3>
        struct tuple_cat_result<T0 , T1 , T2 , T3>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 4>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 4
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 4
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4> >::value == 5
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 5
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    class tuple<T0 , T1 , T2 , T3 , T4>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 5
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4> >
      : boost::mpl::size_t<5>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        4
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T4>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T4
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m4._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 &>(
                v0 , v1 , v2 , v3 , v4
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type
              , T4
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 5>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 5
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 5
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3))
              , std::forward<T4>
                    (t4)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5> >::value == 6
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 6
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 6
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5> >
      : boost::mpl::size_t<6>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        5
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T5>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T5
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m5._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 &>(
                v0 , v1 , v2 , v3 , v4 , v5
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 6>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 6
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 6
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >::value == 7
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 7
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 7
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6> >
      : boost::mpl::size_t<7>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        6
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T6>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T6
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m6._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type
              , T6
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 7>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 7
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 7
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5))
              , std::forward<T6>
                    (t6)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >::value == 8
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 8
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 8
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7> >
      : boost::mpl::size_t<8>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        7
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T7>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T7
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m7._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 8>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 8
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 8
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> >::value == 9
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 9
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 9
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8> >
      : boost::mpl::size_t<9>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        8
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T8>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T8
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m8._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type
              , T8
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 9>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 9
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 9
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7))
              , std::forward<T8>
                    (t8)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> >::value == 10
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 10
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 10
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9> >
      : boost::mpl::size_t<10>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        9
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T9>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T9
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m9._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 10>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 10
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 10
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> >::value == 11
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 11
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 11
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10> >
      : boost::mpl::size_t<11>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        10
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T10>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T10
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m10._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type
              , T10
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 11>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 11
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 11
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9))
              , std::forward<T10>
                    (t10)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> >::value == 12
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 12
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 12
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11> >
      : boost::mpl::size_t<12>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        11
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T11>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T11
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m11._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 12>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 12
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 12
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> >::value == 13
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 13
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 13
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12> >
      : boost::mpl::size_t<13>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        12
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T12>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T12
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m12._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type
              , T12
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 13>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 13
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 13
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11))
              , std::forward<T12>
                    (t12)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13> >::value == 14
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 14
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()) , util::get< 13>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12; detail::tuple_member<T13> _m13;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12() , _m13()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12) , _m13(v13)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12)) , _m13 (std::forward<U13>(u13))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12) , _m13(other._m13)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12)) , _m13(std::move(other._m13))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other))) , _m13(util::get< 13>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value )) && BOOST_NOEXCEPT_EXPR(( _m13._value = other._m13._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value; _m13._value = other._m13._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = std::forward<T13> (other._m13._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value); _m13._value = std::forward<T13> (other._m13._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 14
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = util::get< 13>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other)); _m13._value = util::get< 13>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m13._value , other._m13._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value ); boost::swap( _m13._value , other._m13._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13> >
      : boost::mpl::size_t<14>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        13
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T13>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T13
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m13._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12 , T13 & v13) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type , typename tuple_cat_result< T12 , T13 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 14>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type , typename tuple_element< 13, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 14
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t)) , util::get< 13>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 14
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 13 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type , typename boost::remove_reference<T13>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11)) , util::tuple_cat( std::forward<T12> (t12) , std::forward<T13> (t13))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14> >::value == 15
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 15
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()) , util::get< 13>(boost::declval<UTuple>()) , util::get< 14>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12; detail::tuple_member<T13> _m13; detail::tuple_member<T14> _m14;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12() , _m13() , _m14()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12) , _m13(v13) , _m14(v14)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12)) , _m13 (std::forward<U13>(u13)) , _m14 (std::forward<U14>(u14))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12) , _m13(other._m13) , _m14(other._m14)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12)) , _m13(std::move(other._m13)) , _m14(std::move(other._m14))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other))) , _m13(util::get< 13>(std::forward<UTuple>(other))) , _m14(util::get< 14>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value )) && BOOST_NOEXCEPT_EXPR(( _m13._value = other._m13._value )) && BOOST_NOEXCEPT_EXPR(( _m14._value = other._m14._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value; _m13._value = other._m13._value; _m14._value = other._m14._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = std::forward<T13> (other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = std::forward<T14> (other._m14._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value); _m13._value = std::forward<T13> (other._m13._value); _m14._value = std::forward<T14> (other._m14._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 15
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = util::get< 13>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = util::get< 14>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other)); _m13._value = util::get< 13>(std::forward<UTuple>(other)); _m14._value = util::get< 14>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m13._value , other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m14._value , other._m14._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value ); boost::swap( _m13._value , other._m13._value ); boost::swap( _m14._value , other._m14._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14> >
      : boost::mpl::size_t<15>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        14
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T14>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T14
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m14._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12 , T13 & v13 , T14 & v14) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type , typename tuple_cat_result< T12 , T13 >::type
              , T14
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 15>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type , typename tuple_element< 13, Tuple>::type , typename tuple_element< 14, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 15
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t)) , util::get< 13>(std::forward<Tuple>(t)) , util::get< 14>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 15
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 13 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 14 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type , typename boost::remove_reference<T13>::type , typename boost::remove_reference<T14>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11)) , util::tuple_cat( std::forward<T12> (t12) , std::forward<T13> (t13))
              , std::forward<T14>
                    (t14)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15> >::value == 16
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 16
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()) , util::get< 13>(boost::declval<UTuple>()) , util::get< 14>(boost::declval<UTuple>()) , util::get< 15>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12; detail::tuple_member<T13> _m13; detail::tuple_member<T14> _m14; detail::tuple_member<T15> _m15;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12() , _m13() , _m14() , _m15()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12) , _m13(v13) , _m14(v14) , _m15(v15)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12)) , _m13 (std::forward<U13>(u13)) , _m14 (std::forward<U14>(u14)) , _m15 (std::forward<U15>(u15))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12) , _m13(other._m13) , _m14(other._m14) , _m15(other._m15)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12)) , _m13(std::move(other._m13)) , _m14(std::move(other._m14)) , _m15(std::move(other._m15))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other))) , _m13(util::get< 13>(std::forward<UTuple>(other))) , _m14(util::get< 14>(std::forward<UTuple>(other))) , _m15(util::get< 15>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value )) && BOOST_NOEXCEPT_EXPR(( _m13._value = other._m13._value )) && BOOST_NOEXCEPT_EXPR(( _m14._value = other._m14._value )) && BOOST_NOEXCEPT_EXPR(( _m15._value = other._m15._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value; _m13._value = other._m13._value; _m14._value = other._m14._value; _m15._value = other._m15._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = std::forward<T13> (other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = std::forward<T14> (other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = std::forward<T15> (other._m15._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value); _m13._value = std::forward<T13> (other._m13._value); _m14._value = std::forward<T14> (other._m14._value); _m15._value = std::forward<T15> (other._m15._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 16
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = util::get< 13>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = util::get< 14>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = util::get< 15>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other)); _m13._value = util::get< 13>(std::forward<UTuple>(other)); _m14._value = util::get< 14>(std::forward<UTuple>(other)); _m15._value = util::get< 15>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m13._value , other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m14._value , other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m15._value , other._m15._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value ); boost::swap( _m13._value , other._m13._value ); boost::swap( _m14._value , other._m14._value ); boost::swap( _m15._value , other._m15._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15> >
      : boost::mpl::size_t<16>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        15
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T15>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T15
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m15._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12 , T13 & v13 , T14 & v14 , T15 & v15) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type , typename tuple_cat_result< T12 , T13 >::type , typename tuple_cat_result< T14 , T15 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 16>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type , typename tuple_element< 13, Tuple>::type , typename tuple_element< 14, Tuple>::type , typename tuple_element< 15, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 16
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t)) , util::get< 13>(std::forward<Tuple>(t)) , util::get< 14>(std::forward<Tuple>(t)) , util::get< 15>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 16
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 13 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 14 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 15 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type , typename boost::remove_reference<T13>::type , typename boost::remove_reference<T14>::type , typename boost::remove_reference<T15>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11)) , util::tuple_cat( std::forward<T12> (t12) , std::forward<T13> (t13)) , util::tuple_cat( std::forward<T14> (t14) , std::forward<T15> (t15))
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16> >::value == 17
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 17
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()) , util::get< 13>(boost::declval<UTuple>()) , util::get< 14>(boost::declval<UTuple>()) , util::get< 15>(boost::declval<UTuple>()) , util::get< 16>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    class tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16>
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12; detail::tuple_member<T13> _m13; detail::tuple_member<T14> _m14; detail::tuple_member<T15> _m15; detail::tuple_member<T16> _m16;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12() , _m13() , _m14() , _m15() , _m16()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12) , _m13(v13) , _m14(v14) , _m15(v15) , _m16(v16)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12)) , _m13 (std::forward<U13>(u13)) , _m14 (std::forward<U14>(u14)) , _m15 (std::forward<U15>(u15)) , _m16 (std::forward<U16>(u16))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12) , _m13(other._m13) , _m14(other._m14) , _m15(other._m15) , _m16(other._m16)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12)) , _m13(std::move(other._m13)) , _m14(std::move(other._m14)) , _m15(std::move(other._m15)) , _m16(std::move(other._m16))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other))) , _m13(util::get< 13>(std::forward<UTuple>(other))) , _m14(util::get< 14>(std::forward<UTuple>(other))) , _m15(util::get< 15>(std::forward<UTuple>(other))) , _m16(util::get< 16>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value )) && BOOST_NOEXCEPT_EXPR(( _m13._value = other._m13._value )) && BOOST_NOEXCEPT_EXPR(( _m14._value = other._m14._value )) && BOOST_NOEXCEPT_EXPR(( _m15._value = other._m15._value )) && BOOST_NOEXCEPT_EXPR(( _m16._value = other._m16._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value; _m13._value = other._m13._value; _m14._value = other._m14._value; _m15._value = other._m15._value; _m16._value = other._m16._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = std::forward<T13> (other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = std::forward<T14> (other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = std::forward<T15> (other._m15._value) )) && BOOST_NOEXCEPT_EXPR(( _m16._value = std::forward<T16> (other._m16._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value); _m13._value = std::forward<T13> (other._m13._value); _m14._value = std::forward<T14> (other._m14._value); _m15._value = std::forward<T15> (other._m15._value); _m16._value = std::forward<T16> (other._m16._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 17
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = util::get< 13>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = util::get< 14>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = util::get< 15>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m16._value = util::get< 16>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other)); _m13._value = util::get< 13>(std::forward<UTuple>(other)); _m14._value = util::get< 14>(std::forward<UTuple>(other)); _m15._value = util::get< 15>(std::forward<UTuple>(other)); _m16._value = util::get< 16>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m13._value , other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m14._value , other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m15._value , other._m15._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m16._value , other._m16._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value ); boost::swap( _m13._value , other._m13._value ); boost::swap( _m14._value , other._m14._value ); boost::swap( _m15._value , other._m15._value ); boost::swap( _m16._value , other._m16._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16> >
      : boost::mpl::size_t<17>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        16
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T16>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T16
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m16._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type , typename detail::make_tuple_element<T16>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type , typename detail::make_tuple_element<T16>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 && , T16 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 && , T16 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 & , T16 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12 , T13 & v13 , T14 & v14 , T15 & v15 , T16 & v16) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 & , T16 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type , typename tuple_cat_result< T12 , T13 >::type , typename tuple_cat_result< T14 , T15 >::type
              , T16
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 17>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type , typename tuple_element< 13, Tuple>::type , typename tuple_element< 14, Tuple>::type , typename tuple_element< 15, Tuple>::type , typename tuple_element< 16, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 17
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t)) , util::get< 13>(std::forward<Tuple>(t)) , util::get< 14>(std::forward<Tuple>(t)) , util::get< 15>(std::forward<Tuple>(t)) , util::get< 16>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 17
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 13 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 14 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 15 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 16 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type , typename boost::remove_reference<T13>::type , typename boost::remove_reference<T14>::type , typename boost::remove_reference<T15>::type , typename boost::remove_reference<T16>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11)) , util::tuple_cat( std::forward<T12> (t12) , std::forward<T13> (t13)) , util::tuple_cat( std::forward<T14> (t14) , std::forward<T15> (t15))
              , std::forward<T16>
                    (t16)
            );
    }
}}
namespace hpx { namespace util
{
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17, typename UTuple>
        struct are_tuples_compatible<
            tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>, UTuple
          , typename boost::enable_if_c<
                tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17> >::value == 18
             && tuple_size<typename boost::remove_reference<UTuple>::type>::value == 18
            >::type
        >
        {
            typedef char(&no_type)[1];
            typedef char(&yes_type)[2];
            static no_type call(...);
            static yes_type call(T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17);
            static bool const value =
                sizeof(
                    call(util::get< 0>(boost::declval<UTuple>()) , util::get< 1>(boost::declval<UTuple>()) , util::get< 2>(boost::declval<UTuple>()) , util::get< 3>(boost::declval<UTuple>()) , util::get< 4>(boost::declval<UTuple>()) , util::get< 5>(boost::declval<UTuple>()) , util::get< 6>(boost::declval<UTuple>()) , util::get< 7>(boost::declval<UTuple>()) , util::get< 8>(boost::declval<UTuple>()) , util::get< 9>(boost::declval<UTuple>()) , util::get< 10>(boost::declval<UTuple>()) , util::get< 11>(boost::declval<UTuple>()) , util::get< 12>(boost::declval<UTuple>()) , util::get< 13>(boost::declval<UTuple>()) , util::get< 14>(boost::declval<UTuple>()) , util::get< 15>(boost::declval<UTuple>()) , util::get< 16>(boost::declval<UTuple>()) , util::get< 17>(boost::declval<UTuple>()))
                ) == sizeof(yes_type);
            typedef boost::mpl::bool_<value> type;
        };
    }
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    class tuple
    {
    public: 
        detail::tuple_member<T0> _m0; detail::tuple_member<T1> _m1; detail::tuple_member<T2> _m2; detail::tuple_member<T3> _m3; detail::tuple_member<T4> _m4; detail::tuple_member<T5> _m5; detail::tuple_member<T6> _m6; detail::tuple_member<T7> _m7; detail::tuple_member<T8> _m8; detail::tuple_member<T9> _m9; detail::tuple_member<T10> _m10; detail::tuple_member<T11> _m11; detail::tuple_member<T12> _m12; detail::tuple_member<T13> _m13; detail::tuple_member<T14> _m14; detail::tuple_member<T15> _m15; detail::tuple_member<T16> _m16; detail::tuple_member<T17> _m17;;
    public:
        
        
        
        BOOST_CONSTEXPR tuple()
          : _m0() , _m1() , _m2() , _m3() , _m4() , _m5() , _m6() , _m7() , _m8() , _m9() , _m10() , _m11() , _m12() , _m13() , _m14() , _m15() , _m16() , _m17()
        {}
        
        
        
        BOOST_CONSTEXPR explicit tuple(
            T0 const& v0 , T1 const& v1 , T2 const& v2 , T3 const& v3 , T4 const& v4 , T5 const& v5 , T6 const& v6 , T7 const& v7 , T8 const& v8 , T9 const& v9 , T10 const& v10 , T11 const& v11 , T12 const& v12 , T13 const& v13 , T14 const& v14 , T15 const& v15 , T16 const& v16 , T17 const& v17
        ) : _m0(v0) , _m1(v1) , _m2(v2) , _m3(v3) , _m4(v4) , _m5(v5) , _m6(v6) , _m7(v7) , _m8(v8) , _m9(v9) , _m10(v10) , _m11(v11) , _m12(v12) , _m13(v13) , _m14(v14) , _m15(v15) , _m16(v16) , _m17(v17)
        {}
        
        
        
        
        
        
        
        template <typename U0 , typename U1 , typename U2 , typename U3 , typename U4 , typename U5 , typename U6 , typename U7 , typename U8 , typename U9 , typename U10 , typename U11 , typename U12 , typename U13 , typename U14 , typename U15 , typename U16 , typename U17>
        BOOST_CONSTEXPR explicit tuple(
            U0 && u0 , U1 && u1 , U2 && u2 , U3 && u3 , U4 && u4 , U5 && u5 , U6 && u6 , U7 && u7 , U8 && u8 , U9 && u9 , U10 && u10 , U11 && u11 , U12 && u12 , U13 && u13 , U14 && u14 , U15 && u15 , U16 && u16 , U17 && u17
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<
                    tuple
                  , tuple<U0 , U1 , U2 , U3 , U4 , U5 , U6 , U7 , U8 , U9 , U10 , U11 , U12 , U13 , U14 , U15 , U16 , U17>&&
                >::value
            >::type* = 0
        ) : _m0 (std::forward<U0>(u0)) , _m1 (std::forward<U1>(u1)) , _m2 (std::forward<U2>(u2)) , _m3 (std::forward<U3>(u3)) , _m4 (std::forward<U4>(u4)) , _m5 (std::forward<U5>(u5)) , _m6 (std::forward<U6>(u6)) , _m7 (std::forward<U7>(u7)) , _m8 (std::forward<U8>(u8)) , _m9 (std::forward<U9>(u9)) , _m10 (std::forward<U10>(u10)) , _m11 (std::forward<U11>(u11)) , _m12 (std::forward<U12>(u12)) , _m13 (std::forward<U13>(u13)) , _m14 (std::forward<U14>(u14)) , _m15 (std::forward<U15>(u15)) , _m16 (std::forward<U16>(u16)) , _m17 (std::forward<U17>(u17))
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple const& other)
          : _m0(other._m0) , _m1(other._m1) , _m2(other._m2) , _m3(other._m3) , _m4(other._m4) , _m5(other._m5) , _m6(other._m6) , _m7(other._m7) , _m8(other._m8) , _m9(other._m9) , _m10(other._m10) , _m11(other._m11) , _m12(other._m12) , _m13(other._m13) , _m14(other._m14) , _m15(other._m15) , _m16(other._m16) , _m17(other._m17)
        {}
        
        
        
        BOOST_CONSTEXPR tuple(tuple && other)
          : _m0(std::move(other._m0)) , _m1(std::move(other._m1)) , _m2(std::move(other._m2)) , _m3(std::move(other._m3)) , _m4(std::move(other._m4)) , _m5(std::move(other._m5)) , _m6(std::move(other._m6)) , _m7(std::move(other._m7)) , _m8(std::move(other._m8)) , _m9(std::move(other._m9)) , _m10(std::move(other._m10)) , _m11(std::move(other._m11)) , _m12(std::move(other._m12)) , _m13(std::move(other._m13)) , _m14(std::move(other._m14)) , _m15(std::move(other._m15)) , _m16(std::move(other._m16)) , _m17(std::move(other._m17))
        {}
        
        
        
        
        
        
        
        template <typename UTuple>
        BOOST_CONSTEXPR tuple(
            UTuple && other
          , typename boost::enable_if_c<
                detail::are_tuples_compatible<tuple, UTuple&&>::value
            >::type* = 0
        ) : _m0(util::get< 0>(std::forward<UTuple>(other))) , _m1(util::get< 1>(std::forward<UTuple>(other))) , _m2(util::get< 2>(std::forward<UTuple>(other))) , _m3(util::get< 3>(std::forward<UTuple>(other))) , _m4(util::get< 4>(std::forward<UTuple>(other))) , _m5(util::get< 5>(std::forward<UTuple>(other))) , _m6(util::get< 6>(std::forward<UTuple>(other))) , _m7(util::get< 7>(std::forward<UTuple>(other))) , _m8(util::get< 8>(std::forward<UTuple>(other))) , _m9(util::get< 9>(std::forward<UTuple>(other))) , _m10(util::get< 10>(std::forward<UTuple>(other))) , _m11(util::get< 11>(std::forward<UTuple>(other))) , _m12(util::get< 12>(std::forward<UTuple>(other))) , _m13(util::get< 13>(std::forward<UTuple>(other))) , _m14(util::get< 14>(std::forward<UTuple>(other))) , _m15(util::get< 15>(std::forward<UTuple>(other))) , _m16(util::get< 16>(std::forward<UTuple>(other))) , _m17(util::get< 17>(std::forward<UTuple>(other)))
        {}
        
        
        
        tuple& operator=(tuple const& other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = other._m0._value )) && BOOST_NOEXCEPT_EXPR(( _m1._value = other._m1._value )) && BOOST_NOEXCEPT_EXPR(( _m2._value = other._m2._value )) && BOOST_NOEXCEPT_EXPR(( _m3._value = other._m3._value )) && BOOST_NOEXCEPT_EXPR(( _m4._value = other._m4._value )) && BOOST_NOEXCEPT_EXPR(( _m5._value = other._m5._value )) && BOOST_NOEXCEPT_EXPR(( _m6._value = other._m6._value )) && BOOST_NOEXCEPT_EXPR(( _m7._value = other._m7._value )) && BOOST_NOEXCEPT_EXPR(( _m8._value = other._m8._value )) && BOOST_NOEXCEPT_EXPR(( _m9._value = other._m9._value )) && BOOST_NOEXCEPT_EXPR(( _m10._value = other._m10._value )) && BOOST_NOEXCEPT_EXPR(( _m11._value = other._m11._value )) && BOOST_NOEXCEPT_EXPR(( _m12._value = other._m12._value )) && BOOST_NOEXCEPT_EXPR(( _m13._value = other._m13._value )) && BOOST_NOEXCEPT_EXPR(( _m14._value = other._m14._value )) && BOOST_NOEXCEPT_EXPR(( _m15._value = other._m15._value )) && BOOST_NOEXCEPT_EXPR(( _m16._value = other._m16._value )) && BOOST_NOEXCEPT_EXPR(( _m17._value = other._m17._value ))
            )
        {
            _m0._value = other._m0._value; _m1._value = other._m1._value; _m2._value = other._m2._value; _m3._value = other._m3._value; _m4._value = other._m4._value; _m5._value = other._m5._value; _m6._value = other._m6._value; _m7._value = other._m7._value; _m8._value = other._m8._value; _m9._value = other._m9._value; _m10._value = other._m10._value; _m11._value = other._m11._value; _m12._value = other._m12._value; _m13._value = other._m13._value; _m14._value = other._m14._value; _m15._value = other._m15._value; _m16._value = other._m16._value; _m17._value = other._m17._value;;
            return *this;
        }
        
        
        tuple& operator=(tuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = std::forward<T0> (other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = std::forward<T1> (other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = std::forward<T2> (other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = std::forward<T3> (other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = std::forward<T4> (other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = std::forward<T5> (other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = std::forward<T6> (other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = std::forward<T7> (other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = std::forward<T8> (other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = std::forward<T9> (other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = std::forward<T10> (other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = std::forward<T11> (other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = std::forward<T12> (other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = std::forward<T13> (other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = std::forward<T14> (other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = std::forward<T15> (other._m15._value) )) && BOOST_NOEXCEPT_EXPR(( _m16._value = std::forward<T16> (other._m16._value) )) && BOOST_NOEXCEPT_EXPR(( _m17._value = std::forward<T17> (other._m17._value) ))
            )
        {
            _m0._value = std::forward<T0> (other._m0._value); _m1._value = std::forward<T1> (other._m1._value); _m2._value = std::forward<T2> (other._m2._value); _m3._value = std::forward<T3> (other._m3._value); _m4._value = std::forward<T4> (other._m4._value); _m5._value = std::forward<T5> (other._m5._value); _m6._value = std::forward<T6> (other._m6._value); _m7._value = std::forward<T7> (other._m7._value); _m8._value = std::forward<T8> (other._m8._value); _m9._value = std::forward<T9> (other._m9._value); _m10._value = std::forward<T10> (other._m10._value); _m11._value = std::forward<T11> (other._m11._value); _m12._value = std::forward<T12> (other._m12._value); _m13._value = std::forward<T13> (other._m13._value); _m14._value = std::forward<T14> (other._m14._value); _m15._value = std::forward<T15> (other._m15._value); _m16._value = std::forward<T16> (other._m16._value); _m17._value = std::forward<T17> (other._m17._value);;
            return *this;
        }
        
        
        
        template <typename UTuple>
        typename boost::enable_if_c<
            tuple_size<typename boost::remove_reference<UTuple>::type>::value == 18
          , tuple&
        >::type
        operator=(UTuple && other)
            HPX_UTIL_TUPLE_SFINAE_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( _m0._value = util::get< 0>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m1._value = util::get< 1>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m2._value = util::get< 2>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m3._value = util::get< 3>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m4._value = util::get< 4>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m5._value = util::get< 5>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m6._value = util::get< 6>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m7._value = util::get< 7>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m8._value = util::get< 8>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m9._value = util::get< 9>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m10._value = util::get< 10>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m11._value = util::get< 11>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m12._value = util::get< 12>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m13._value = util::get< 13>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m14._value = util::get< 14>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m15._value = util::get< 15>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m16._value = util::get< 16>(std::forward<UTuple>(other)) )) && BOOST_NOEXCEPT_EXPR(( _m17._value = util::get< 17>(std::forward<UTuple>(other)) ))
            )
        {
            _m0._value = util::get< 0>(std::forward<UTuple>(other)); _m1._value = util::get< 1>(std::forward<UTuple>(other)); _m2._value = util::get< 2>(std::forward<UTuple>(other)); _m3._value = util::get< 3>(std::forward<UTuple>(other)); _m4._value = util::get< 4>(std::forward<UTuple>(other)); _m5._value = util::get< 5>(std::forward<UTuple>(other)); _m6._value = util::get< 6>(std::forward<UTuple>(other)); _m7._value = util::get< 7>(std::forward<UTuple>(other)); _m8._value = util::get< 8>(std::forward<UTuple>(other)); _m9._value = util::get< 9>(std::forward<UTuple>(other)); _m10._value = util::get< 10>(std::forward<UTuple>(other)); _m11._value = util::get< 11>(std::forward<UTuple>(other)); _m12._value = util::get< 12>(std::forward<UTuple>(other)); _m13._value = util::get< 13>(std::forward<UTuple>(other)); _m14._value = util::get< 14>(std::forward<UTuple>(other)); _m15._value = util::get< 15>(std::forward<UTuple>(other)); _m16._value = util::get< 16>(std::forward<UTuple>(other)); _m17._value = util::get< 17>(std::forward<UTuple>(other));;
            return *this;
        }
        
        
        
        
        void swap(tuple& other)
            BOOST_NOEXCEPT_IF(
                true && BOOST_NOEXCEPT_EXPR(( boost::swap( _m0._value , other._m0._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m1._value , other._m1._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m2._value , other._m2._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m3._value , other._m3._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m4._value , other._m4._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m5._value , other._m5._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m6._value , other._m6._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m7._value , other._m7._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m8._value , other._m8._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m9._value , other._m9._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m10._value , other._m10._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m11._value , other._m11._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m12._value , other._m12._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m13._value , other._m13._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m14._value , other._m14._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m15._value , other._m15._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m16._value , other._m16._value) )) && BOOST_NOEXCEPT_EXPR(( boost::swap( _m17._value , other._m17._value) ))
            )
        {
            boost::swap( _m0._value , other._m0._value ); boost::swap( _m1._value , other._m1._value ); boost::swap( _m2._value , other._m2._value ); boost::swap( _m3._value , other._m3._value ); boost::swap( _m4._value , other._m4._value ); boost::swap( _m5._value , other._m5._value ); boost::swap( _m6._value , other._m6._value ); boost::swap( _m7._value , other._m7._value ); boost::swap( _m8._value , other._m8._value ); boost::swap( _m9._value , other._m9._value ); boost::swap( _m10._value , other._m10._value ); boost::swap( _m11._value , other._m11._value ); boost::swap( _m12._value , other._m12._value ); boost::swap( _m13._value , other._m13._value ); boost::swap( _m14._value , other._m14._value ); boost::swap( _m15._value , other._m15._value ); boost::swap( _m16._value , other._m16._value ); boost::swap( _m17._value , other._m17._value );;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_size<tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17> >
      : boost::mpl::size_t<18>
    {};
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    struct tuple_element<
        17
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
    > : boost::mpl::identity<T17>
    {
        template <typename Tuple>
        static BOOST_CONSTEXPR BOOST_FORCEINLINE
        typename detail::qualify_as<
            T17
          , Tuple&
        >::type
        get(Tuple& tuple) BOOST_NOEXCEPT
        {
            return tuple._m17._value;
        }
    };
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type , typename detail::make_tuple_element<T16>::type , typename detail::make_tuple_element<T17>::type>
    make_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17)
    {
        return
            tuple<typename detail::make_tuple_element<T0>::type , typename detail::make_tuple_element<T1>::type , typename detail::make_tuple_element<T2>::type , typename detail::make_tuple_element<T3>::type , typename detail::make_tuple_element<T4>::type , typename detail::make_tuple_element<T5>::type , typename detail::make_tuple_element<T6>::type , typename detail::make_tuple_element<T7>::type , typename detail::make_tuple_element<T8>::type , typename detail::make_tuple_element<T9>::type , typename detail::make_tuple_element<T10>::type , typename detail::make_tuple_element<T11>::type , typename detail::make_tuple_element<T12>::type , typename detail::make_tuple_element<T13>::type , typename detail::make_tuple_element<T14>::type , typename detail::make_tuple_element<T15>::type , typename detail::make_tuple_element<T16>::type , typename detail::make_tuple_element<T17>::type>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 )
            );
    }
    
    
    
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    BOOST_FORCEINLINE
    tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 && , T16 && , T17 &&>
    forward_as_tuple(T0 && v0 , T1 && v1 , T2 && v2 , T3 && v3 , T4 && v4 , T5 && v5 , T6 && v6 , T7 && v7 , T8 && v8 , T9 && v9 , T10 && v10 , T11 && v11 , T12 && v12 , T13 && v13 , T14 && v14 , T15 && v15 , T16 && v16 , T17 && v17) BOOST_NOEXCEPT
    {
        return
            tuple<T0 && , T1 && , T2 && , T3 && , T4 && , T5 && , T6 && , T7 && , T8 && , T9 && , T10 && , T11 && , T12 && , T13 && , T14 && , T15 && , T16 && , T17 &&>(
                std::forward<T0>( v0 ) , std::forward<T1>( v1 ) , std::forward<T2>( v2 ) , std::forward<T3>( v3 ) , std::forward<T4>( v4 ) , std::forward<T5>( v5 ) , std::forward<T6>( v6 ) , std::forward<T7>( v7 ) , std::forward<T8>( v8 ) , std::forward<T9>( v9 ) , std::forward<T10>( v10 ) , std::forward<T11>( v11 ) , std::forward<T12>( v12 ) , std::forward<T13>( v13 ) , std::forward<T14>( v14 ) , std::forward<T15>( v15 ) , std::forward<T16>( v16 ) , std::forward<T17>( v17 )
            );
    }
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    BOOST_FORCEINLINE
    tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 & , T16 & , T17 &>
    tie(T0 & v0 , T1 & v1 , T2 & v2 , T3 & v3 , T4 & v4 , T5 & v5 , T6 & v6 , T7 & v7 , T8 & v8 , T9 & v9 , T10 & v10 , T11 & v11 , T12 & v12 , T13 & v13 , T14 & v14 , T15 & v15 , T16 & v16 , T17 & v17) BOOST_NOEXCEPT
    {
        return
            tuple<T0 & , T1 & , T2 & , T3 & , T4 & , T5 & , T6 & , T7 & , T8 & , T9 & , T10 & , T11 & , T12 & , T13 & , T14 & , T15 & , T16 & , T17 &>(
                v0 , v1 , v2 , v3 , v4 , v5 , v6 , v7 , v8 , v9 , v10 , v11 , v12 , v13 , v14 , v15 , v16 , v17
            );
    }
    
    
    namespace detail
    {
        template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
        struct tuple_cat_result<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12 , T13 , T14 , T15 , T16 , T17>
          : tuple_cat_result<
                typename tuple_cat_result< T0 , T1 >::type , typename tuple_cat_result< T2 , T3 >::type , typename tuple_cat_result< T4 , T5 >::type , typename tuple_cat_result< T6 , T7 >::type , typename tuple_cat_result< T8 , T9 >::type , typename tuple_cat_result< T10 , T11 >::type , typename tuple_cat_result< T12 , T13 >::type , typename tuple_cat_result< T14 , T15 >::type , typename tuple_cat_result< T16 , T17 >::type
            >
        {};
        template <typename Tuple>
        struct tuple_cat_result<
            Tuple
          , typename boost::enable_if_c<tuple_size<Tuple>::value == 18>::type
        >
        {
            typedef
                tuple<typename tuple_element< 0, Tuple>::type , typename tuple_element< 1, Tuple>::type , typename tuple_element< 2, Tuple>::type , typename tuple_element< 3, Tuple>::type , typename tuple_element< 4, Tuple>::type , typename tuple_element< 5, Tuple>::type , typename tuple_element< 6, Tuple>::type , typename tuple_element< 7, Tuple>::type , typename tuple_element< 8, Tuple>::type , typename tuple_element< 9, Tuple>::type , typename tuple_element< 10, Tuple>::type , typename tuple_element< 11, Tuple>::type , typename tuple_element< 12, Tuple>::type , typename tuple_element< 13, Tuple>::type , typename tuple_element< 14, Tuple>::type , typename tuple_element< 15, Tuple>::type , typename tuple_element< 16, Tuple>::type , typename tuple_element< 17, Tuple>::type>
                type;
        };
    }
    template <typename Tuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<Tuple>::type>::value == 18
      , detail::tuple_cat_result<
            typename boost::remove_reference<Tuple>::type
        >
    >::type
    tuple_cat(Tuple && t)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<Tuple>::type
            >::type(
                util::get< 0>(std::forward<Tuple>(t)) , util::get< 1>(std::forward<Tuple>(t)) , util::get< 2>(std::forward<Tuple>(t)) , util::get< 3>(std::forward<Tuple>(t)) , util::get< 4>(std::forward<Tuple>(t)) , util::get< 5>(std::forward<Tuple>(t)) , util::get< 6>(std::forward<Tuple>(t)) , util::get< 7>(std::forward<Tuple>(t)) , util::get< 8>(std::forward<Tuple>(t)) , util::get< 9>(std::forward<Tuple>(t)) , util::get< 10>(std::forward<Tuple>(t)) , util::get< 11>(std::forward<Tuple>(t)) , util::get< 12>(std::forward<Tuple>(t)) , util::get< 13>(std::forward<Tuple>(t)) , util::get< 14>(std::forward<Tuple>(t)) , util::get< 15>(std::forward<Tuple>(t)) , util::get< 16>(std::forward<Tuple>(t)) , util::get< 17>(std::forward<Tuple>(t))
            );
    }
    template <typename TTuple, typename UTuple>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename boost::lazy_enable_if_c<
        tuple_size<typename boost::remove_reference<TTuple>::type>::value
      + tuple_size<typename boost::remove_reference<UTuple>::type>::value == 18
      , detail::tuple_cat_result<
            typename boost::remove_reference<TTuple>::type
          , typename boost::remove_reference<UTuple>::type
        >
    >::type
    tuple_cat(TTuple && t, UTuple && u)
    {
        return
            typename detail::tuple_cat_result<
                typename boost::remove_reference<TTuple>::type
              , typename boost::remove_reference<UTuple>::type
            >::type(
                detail::tuple_cat_element< 0 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 1 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 2 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 3 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 4 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 5 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 6 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 7 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 8 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 9 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 10 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 11 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 12 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 13 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 14 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 15 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 16 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u)) , detail::tuple_cat_element< 17 , typename boost::remove_reference<TTuple>::type , typename boost::remove_reference<UTuple>::type >::call(std::forward<TTuple>(t), std::forward<UTuple>(u))
            );
    }
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12 , typename T13 , typename T14 , typename T15 , typename T16 , typename T17>
    BOOST_CONSTEXPR BOOST_FORCEINLINE
    typename detail::tuple_cat_result<
        typename boost::remove_reference<T0>::type , typename boost::remove_reference<T1>::type , typename boost::remove_reference<T2>::type , typename boost::remove_reference<T3>::type , typename boost::remove_reference<T4>::type , typename boost::remove_reference<T5>::type , typename boost::remove_reference<T6>::type , typename boost::remove_reference<T7>::type , typename boost::remove_reference<T8>::type , typename boost::remove_reference<T9>::type , typename boost::remove_reference<T10>::type , typename boost::remove_reference<T11>::type , typename boost::remove_reference<T12>::type , typename boost::remove_reference<T13>::type , typename boost::remove_reference<T14>::type , typename boost::remove_reference<T15>::type , typename boost::remove_reference<T16>::type , typename boost::remove_reference<T17>::type
    >::type
    tuple_cat(T0 && t0 , T1 && t1 , T2 && t2 , T3 && t3 , T4 && t4 , T5 && t5 , T6 && t6 , T7 && t7 , T8 && t8 , T9 && t9 , T10 && t10 , T11 && t11 , T12 && t12 , T13 && t13 , T14 && t14 , T15 && t15 , T16 && t16 , T17 && t17)
    {
        return
            util::tuple_cat(
                util::tuple_cat( std::forward<T0> (t0) , std::forward<T1> (t1)) , util::tuple_cat( std::forward<T2> (t2) , std::forward<T3> (t3)) , util::tuple_cat( std::forward<T4> (t4) , std::forward<T5> (t5)) , util::tuple_cat( std::forward<T6> (t6) , std::forward<T7> (t7)) , util::tuple_cat( std::forward<T8> (t8) , std::forward<T9> (t9)) , util::tuple_cat( std::forward<T10> (t10) , std::forward<T11> (t11)) , util::tuple_cat( std::forward<T12> (t12) , std::forward<T13> (t13)) , util::tuple_cat( std::forward<T14> (t14) , std::forward<T15> (t15)) , util::tuple_cat( std::forward<T16> (t16) , std::forward<T17> (t17))
            );
    }
}}
