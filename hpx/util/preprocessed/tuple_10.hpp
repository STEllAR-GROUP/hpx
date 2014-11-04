// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


        
namespace hpx { namespace util
{
    
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
                    boost::is_same<
                        tuple, typename boost::remove_reference<U0>::type
                    >::value || detail::are_tuples_compatible_not_same<
                        tuple, U0&&
                    >::value
                  , boost::mpl::false_
                  , detail::are_tuples_compatible<tuple, tuple<U0>&&>
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0, typename ...Tail>
    struct tuple_element<
        0
      , tuple<T0, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1, typename ...Tail>
    struct tuple_element<
        1
      , tuple<T0 , T1, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2, typename ...Tail>
    struct tuple_element<
        2
      , tuple<T0 , T1 , T2, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3, typename ...Tail>
    struct tuple_element<
        3
      , tuple<T0 , T1 , T2 , T3, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4, typename ...Tail>
    struct tuple_element<
        4
      , tuple<T0 , T1 , T2 , T3 , T4, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5, typename ...Tail>
    struct tuple_element<
        5
      , tuple<T0 , T1 , T2 , T3 , T4 , T5, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6, typename ...Tail>
    struct tuple_element<
        6
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7, typename ...Tail>
    struct tuple_element<
        7
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8, typename ...Tail>
    struct tuple_element<
        8
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9, typename ...Tail>
    struct tuple_element<
        9
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10, typename ...Tail>
    struct tuple_element<
        10
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11, typename ...Tail>
    struct tuple_element<
        11
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11, Tail...>
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
}}
namespace hpx { namespace util
{
    
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
                detail::are_tuples_compatible_not_same<tuple, UTuple&&>::value
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
    
    
    
    template <typename T0 , typename T1 , typename T2 , typename T3 , typename T4 , typename T5 , typename T6 , typename T7 , typename T8 , typename T9 , typename T10 , typename T11 , typename T12, typename ...Tail>
    struct tuple_element<
        12
      , tuple<T0 , T1 , T2 , T3 , T4 , T5 , T6 , T7 , T8 , T9 , T10 , T11 , T12, Tail...>
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
}}
