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
namespace hpx { namespace util { namespace detail
{
    
    template <typename VTablePtr,
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ));
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
        typename R, typename A0 , typename A1 , typename A2 , typename A3 , typename A4 , typename A5 , typename A6 , typename A7 , typename A8 , typename A9 , typename A10 , typename A11 , typename A12 , typename A13 , typename A14 , typename A15 , typename A16 , typename A17>
    class basic_function<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)>
      : public function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)>
    {
        HPX_MOVABLE_BUT_NOT_COPYABLE(basic_function);
        typedef function_base<VTablePtr, R(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)> base_type;
    public:
        typedef R result_type;
        template <typename T>
        struct is_callable
          : traits::is_callable<T(A0 , A1 , A2 , A3 , A4 , A5 , A6 , A7 , A8 , A9 , A10 , A11 , A12 , A13 , A14 , A15 , A16 , A17)>
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
        BOOST_FORCEINLINE R operator()(A0 a0 , A1 a1 , A2 a2 , A3 a3 , A4 a4 , A5 a5 , A6 a6 , A7 a7 , A8 a8 , A9 a9 , A10 a10 , A11 a11 , A12 a12 , A13 a13 , A14 a14 , A15 a15 , A16 a16 , A17 a17) const
        {
            return this->vptr->invoke(&this->object,
                std::forward<A0>( a0 ) , std::forward<A1>( a1 ) , std::forward<A2>( a2 ) , std::forward<A3>( a3 ) , std::forward<A4>( a4 ) , std::forward<A5>( a5 ) , std::forward<A6>( a6 ) , std::forward<A7>( a7 ) , std::forward<A8>( a8 ) , std::forward<A9>( a9 ) , std::forward<A10>( a10 ) , std::forward<A11>( a11 ) , std::forward<A12>( a12 ) , std::forward<A13>( a13 ) , std::forward<A14>( a14 ) , std::forward<A15>( a15 ) , std::forward<A16>( a16 ) , std::forward<A17>( a17 ));
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
