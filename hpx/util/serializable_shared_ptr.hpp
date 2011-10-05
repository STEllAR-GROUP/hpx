////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_38A9E66A_6CFC_476B_96FE_5DD84E5A4A98)
#define HPX_38A9E66A_6CFC_476B_96FE_5DD84E5A4A98

#include <boost/serialization/access.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/shared_ptr.hpp>

#include <hpx/util/safe_bool.hpp>

namespace hpx { namespace util
{

/// Wrapper to avoid Boost.Serialization warnings about serializing
/// boost::shared_ptr<>s.
template <
    typename T
>
struct serializable_shared_ptr
{
    typedef typename boost::shared_ptr<T>::element_type element_type;

    serializable_shared_ptr()
      : ptr_()
    {}

    template <
        typename Y
    >
    explicit serializable_shared_ptr(
        Y* p
        )
      : ptr_(p)
    {}

    template <
        typename Y
      , typename D
    >
    serializable_shared_ptr(
        Y* p
      , D d
        )
      : ptr_(p, d)
    {}

    template <
        typename Y
      , typename D
      , typename A
    >
    serializable_shared_ptr(
        Y* p
      , D d
      , A a
        )
      : ptr_(p, d, a)
    {}

    serializable_shared_ptr(
        boost::shared_ptr<T> const& r
        )
      : ptr_(r)
    {}

    serializable_shared_ptr(
        serializable_shared_ptr<T> const& r
        )
      : ptr_(r.ptr_)
    {}

    template <
        typename Y
    >
    serializable_shared_ptr(
        boost::shared_ptr<Y> const& r
        )
      : ptr_(r)
    {}

    template <
        typename Y
    >
    serializable_shared_ptr(
        serializable_shared_ptr<Y> const& r
        )
      : ptr_(r.ptr_)
    {}

    template <
        typename Y
    >
    serializable_shared_ptr(
        boost::shared_ptr<Y> const& r
      , T* p
        )
      : ptr_(r, p)
    {}

    template <
        typename Y
    >
    serializable_shared_ptr(
        serializable_shared_ptr<Y> const& r
      , T* p
        )
      : ptr_(r.ptr_, p)
    {}

    template <
        typename Y
    >
    explicit serializable_shared_ptr(
        boost::weak_ptr<Y> const& r
        )
      : ptr_(r)
    {}

    serializable_shared_ptr & operator=(
        boost::shared_ptr<T> const& r
        )
    {
        ptr_ = r;
    }

    serializable_shared_ptr & operator=(
        serializable_shared_ptr const& r
        )
    {
        ptr_ = r.ptr_;
    }

    template <
        typename Y
    >
    serializable_shared_ptr& operator=(
        boost::shared_ptr<Y> const& r
        )
    {
        ptr_ = r;
    }

    template <
        typename Y
    >
    serializable_shared_ptr& operator=(
        serializable_shared_ptr<Y> const& r
        )
    {
        ptr_ = r.ptr_;
    }

    void reset()
    {
        ptr_.reset();
    }

    template <
        typename Y
    >
    void reset(
        Y* p
        )
    {
        ptr_.reset(p);
    }

    template <
        typename Y
      , typename D
    >
    void reset(
        Y* p
      , D d
        )
    {
        ptr_.reset(p, d);
    }

    template <
        typename Y
      , typename D
      , typename A
    >
    void reset(
        Y* p
      , D d
      , A a
        )
    {
        ptr_.reset(p, d, a);
    }

    template <
        typename Y
    >
    void reset(
        boost::shared_ptr<Y> const& r
      , T* p
        )
    {
        ptr_.reset(r, p);
    }

    template <
        typename Y
    >
    void reset(
        serializable_shared_ptr<Y> const& r
      , T* p
        )
    {
        ptr_.reset(r.ptr_, p);
    }

    T& operator*() const
    {
        return ptr_.operator*();
    }

    T* operator->() const
    {
        return ptr_.operator->();
    }

    T* get() const
    {
        return ptr_.get();
    }

    bool unique() const
    {
        return ptr_.unique();
    }
 
    long use_count() const
    {
        return ptr_.use_count();
    } 

    operator typename safe_bool<serializable_shared_ptr>::result_type() const 
    { 
        return bool(ptr_);
    }

    void swap(serializable_shared_ptr& b)
    {
        ptr_.swap(b.ptr_);
    }

    void swap(boost::shared_ptr<T>& b)
    {
        ptr_.swap(b);
    }

  private:
    boost::shared_ptr<T> ptr_;

    friend class boost::serialization::access;

    template <
        typename Archive
    >
    void save(
        Archive& ar
      , const unsigned int
        ) const
    {
        bool isvalid = ptr_ ? true : false;
        ar << isvalid;

        if (isvalid)
        {
            T const& instance = *ptr_;
            ar << instance;
        }
    }

    template <
        typename Archive
    >
    void load(
        Archive& ar
      , const unsigned int
        )
    {
        bool isvalid;
        ar >> isvalid;

        if (isvalid)
        {
            T instance;
            ar >> instance;
            ptr_.reset(new T(instance));
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

}}

#endif // HPX_38A9E66A_6CFC_476B_96FE_5DD84E5A4A98

