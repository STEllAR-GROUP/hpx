//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_POINTER_WRAPPER_VECTOR_HPP
#define HPX_UTIL_POINTER_WRAPPER_VECTOR_HPP

#include <hpx/config.hpp>
//
#include <vector>
#include <functional>
//
namespace hpx { namespace util { namespace detail
{
// this class looks like a vector, but can be initialized from a pointer and size,
// it is used by the verbs parcelport to pass an rdma memory chunk with received
// data into the decode parcel buffer routines.
// it cannot be resized or changed once created and does not delete the memory it wraps
template<class T>
class pointer_wrapper_vector
{
  public:
    typedef std::function<void(void)> deleter_callback;
    T  *m_array_;
    int m_size_;
    deleter_callback cb_;

    typedef T                                           value_type;
    typedef value_type &                                reference;
    typedef const value_type &                          const_reference;
    typedef typename std::vector<T>::iterator           iterator;
    typedef const typename std::vector<T>::iterator     const_iterator;
    typedef typename std::vector<T>::difference_type    difference_type;
    typedef typename std::vector<T>::size_type          size_type;
    typedef typename std::vector<T>::allocator_type     allocator_type;

    pointer_wrapper_vector() : m_array_(0), m_size_(0) {}
    pointer_wrapper_vector(T* p, std::size_t s, deleter_callback cb) :
        m_array_(p), m_size_(s), cb_(cb) {}

    pointer_wrapper_vector(pointer_wrapper_vector<T>&& other, allocator_type allocator = allocator_type()) :
        m_array_(other.m_array_), m_size_(other.m_size_), cb_(other.cb_)
    {
        other.m_size_  = 0;
        other.m_array_ = 0;
        other.cb_      = NULL;
    }

    ~pointer_wrapper_vector() {
        if (m_array_ && cb_) {
            LOG_DEBUG_MSG("Deleting pointer wrapper, should trigger callback to parcelport memory free");
            cb_();
        }
        else {
            LOG_DEBUG_MSG("============ %%%%%%%%%%%%%");
        }
    }
    size_type size() const { return m_size_; }

    size_type max_size() const { return m_size_; }

    bool empty() const { return m_array_ == NULL; }

    iterator begin() {
      return iterator(&m_array_[0]);
    }

    iterator end() {
      return iterator(&m_array_[m_size_]);
    }

    reference       operator[](std::size_t index)       { return m_array_[index]; }
    const_reference operator[](std::size_t index) const { return m_array_[index]; }

//    const T *operator[](std::size_t index) { return &m_array[index]; }
//    T &operator[](std::size_t index)       { return &m_array[index]; }
//    const T &operator[](std::size_t index) { return &m_array[index]; }

    void push_back(const T &_Val) {}

    void resize(std::size_t s) {}
    void reserve(std::size_t s) {}

  private:
    pointer_wrapper_vector(pointer_wrapper_vector<T> const & other, allocator_type allocator = allocator_type());

  };
}}}

#endif
