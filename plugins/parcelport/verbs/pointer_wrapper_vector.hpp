//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_POINTER_WRAPPER_VECTOR_HPP
#define HPX_UTIL_POINTER_WRAPPER_VECTOR_HPP

#include <hpx/config.hpp>
//
#include <vector>

namespace hpx { namespace util { namespace detail
{

template<class T>
class pointer_wrapper_vector
{
  public:
    T  *m_array_;
    int m_size_;

    typedef T                               value_type;
    typedef value_type &                    reference;
    typedef const value_type &              const_reference;
    typedef typename std::vector<T>::iterator        iterator;
    typedef const typename std::vector<T>::iterator  const_iterator;
    typedef typename std::vector<T>::difference_type difference_type;
    typedef typename std::vector<T>::size_type       size_type;
    typedef typename std::vector<T>::allocator_type       allocator_type;


    pointer_wrapper_vector() : m_array_(0), m_size_(0) {}
    pointer_wrapper_vector(T* p, std::size_t s) : m_array_(p), m_size_(s) {}

    pointer_wrapper_vector(pointer_wrapper_vector<T> const & other, allocator_type allocator = allocator_type()) :
      m_array_(other.m_array_), m_size_(other.m_size_) {}

    size_type size() const { return m_size_; }

    size_type max_size() const { return m_size_; }

    bool empty() const { return m_array_ == NULL; }

    iterator begin() {
      return iterator(&m_array_[0]);
    }

    iterator end() {
      return iterator(&m_array_[m_size_]);
    }

    reference operator[](std::size_t index) { return m_array_[index]; }
    const_reference operator[](std::size_t index) const { return m_array_[index]; }

//    const T *operator[](std::size_t index) { return &m_array[index]; }
//    T &operator[](std::size_t index)       { return &m_array[index]; }
//    const T &operator[](std::size_t index) { return &m_array[index]; }

    void push_back(const T &_Val) {}

    void resize(std::size_t s) {}
    void reserve(std::size_t s) {}
  };
}}}

#endif
