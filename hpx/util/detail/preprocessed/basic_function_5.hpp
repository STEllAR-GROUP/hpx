// Copyright (c) 2007-2013 Hartmut Kaiser
// Copyright (c) 2012-2013 Thomas Heller
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// This file has been automatically generated using the Boost.Wave tool.
// Do not edit manually.


namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0>
    class basic_function<VTablePtr, R(A0)>
      : public function_base<VTablePtr, R(A0)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1>
    class basic_function<VTablePtr, R(A0 , A1)>
      : public function_base<VTablePtr, R(A0 , A1)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2>
    class basic_function<VTablePtr, R(A0 , A1 , A2)>
      : public function_base<VTablePtr, R(A0 , A1 , A2)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7)>
        {};
        basic_function() BOOST_NOEXCEPT
          : base_type()
        {}
        basic_function(basic_function&& other) BOOST_NOEXCEPT
          : base_type(static_cast<base_type&&>(other))
        {}
        basic_function& operator=(basic_function&& other) BOOST_NOEXCEPT
        {
            base_type::operator=(static_cast<base_type&&>(other));
            return *this;
        }
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ));
        }
        template <typename T>
        T* target() BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
        template <typename T>
        T* target() const BOOST_NOEXCEPT
        {
            BOOST_STATIC_ASSERT_MSG(
                is_callable<T>::value
              , "T shall be Callable with the function signature"
            );
            return base_type::template target<T>();
        }
    };
}}}
