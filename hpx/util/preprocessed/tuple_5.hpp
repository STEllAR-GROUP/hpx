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
                boost::mpl::eval_if_c<
                    boost::is_base_of<
                        tuple, typename boost::remove_reference<U0>::type
                    >::value || detail::are_tuples_compatible<tuple, U0&&>::value
                  , boost::mpl::false_
                  , detail::are_tuples_compatible<
                        tuple
                      , tuple<U0>&&
                    >
                >::type::value
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        0
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        1
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        2
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        3
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        4
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        5
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        6
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
    class tuple
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
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7>
    struct tuple_element<
        7
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7>
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
