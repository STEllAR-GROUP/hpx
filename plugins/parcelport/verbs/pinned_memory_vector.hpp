//  Copyright (c) 2015 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_UTIL_PINNED_MEMORY_VECTOR_HPP
#define HPX_UTIL_PINNED_MEMORY_VECTOR_HPP

#include <hpx/config.hpp>
//
#include <vector>
#include <functional>
#include "RdmaLogging.h"
#include "RdmaMemoryPool.h"
//
namespace hpx { namespace util { namespace detail
{
// this class looks like a vector, but can be initialized from a pointer and size,
// it is used by the verbs parcelport to pass an rdma memory chunk with received
// data into the decode parcel buffer routines.
// it cannot be resized or changed once created and does not delete the memory it wraps
template<class T, class Allocator = RdmaMemoryPool>
class pinned_memory_vector
{
  public:
    typedef T                                           value_type;
    typedef value_type &                                reference;
    typedef const value_type &                          const_reference;
    typedef typename std::vector<T>::iterator           iterator;
    typedef const typename std::vector<T>::iterator     const_iterator;
    typedef typename std::vector<T>::difference_type    difference_type;
    typedef typename std::vector<T>::size_type          size_type;
    typedef Allocator                                   allocator_type;

    typedef std::function<void(void)> deleter_callback;
    T                *m_array_;
    int               m_size_;
    deleter_callback  m_cb_;
    allocator_type   *m_alloc_;
    RdmaMemoryRegion *m_region_;

    // construct with a memory pool pointer
    pinned_memory_vector(allocator_type* alloc) :
        m_array_(0), m_size_(0), m_cb_(0), m_alloc_(alloc), m_region_(0)
    {
        LOG_DEBUG_MSG("pinned_memory_vector constructed (alloc)" << hexuint32(m_size_) << hexpointer(m_array_) << hexpointer(m_region_));
    }

    // construct from existing memory chunk
    pinned_memory_vector(T* p, std::size_t s, deleter_callback cb, allocator_type* alloc, RdmaMemoryRegion *r) :
        m_array_(p), m_size_(s), m_cb_(cb), m_alloc_(alloc), m_region_(r)
    {
        LOG_DEBUG_MSG("pinned_memory_vector constructed " << hexuint32(m_size_) << hexpointer(m_array_) << hexpointer(m_region_));
    }

    // move constructor, 
    pinned_memory_vector(pinned_memory_vector<T>&& other) :
        m_array_(other.m_array_), m_size_(other.m_size_), 
        m_cb_(other.m_cb_), m_alloc_(std::move(other.m_alloc_)), m_region_(other.m_region_)
    {
        LOG_DEBUG_MSG("pinned_memory_vector moved into " << hexuint32(m_size_) << hexpointer(m_array_) << hexpointer(m_region_));
        other.m_size_    = 0;
        other.m_array_   = 0;
        other.m_cb_      = nullptr;
        other.m_alloc_   = nullptr;
        other.m_region_  = nullptr;
    }

    ~pinned_memory_vector() {
        if (m_array_ && m_cb_) {
            LOG_DEBUG_MSG("pinned_memory_vector calling delete callback, " << hexuint32(m_size_) << hexpointer(m_array_) << hexpointer(m_region_));
            m_cb_();
        }
    }

    // move copy operator
    pinned_memory_vector & operator=(pinned_memory_vector && other)
    {
        m_array_  = other.m_array_;
        m_size_   = other.m_size_;
        m_cb_     = other.m_cb_;
        m_alloc_  = other.m_alloc_;
        m_region_ = other.m_region_;
        LOG_DEBUG_MSG("pinned_memory_vector copied into " << hexuint32(m_size_) << hexpointer(m_array_) << hexpointer(m_region_));
        other.m_size_   = 0;
        other.m_array_  = 0;
        other.m_cb_     = nullptr;
        other.m_alloc_  = nullptr;
        other.m_region_ = nullptr;
        return *this;
    }

    size_type size() const { return m_size_; }

    size_type max_size() const { return m_size_; }

    bool empty() const { return m_array_ == nullptr; }

    T *data() { return m_array_; }

    iterator begin() {
      return iterator(&m_array_[0]);
    }

    iterator end() {
      return iterator(&m_array_[m_size_]);
    }

    const_iterator begin() const {
      return iterator(&m_array_[0]);
    }

    const iterator end() const {
      return iterator(&m_array_[m_size_]);
    }

    reference       operator[](std::size_t index)       { return m_array_[index]; }
    const_reference operator[](std::size_t index) const { return m_array_[index]; }

    void push_back(const T &_Val) {}

    std::size_t capacity() { return m_region_ ? m_region_->getLength() : 0; }

    inline void resize(std::size_t s) {
        LOG_DEBUG_MSG("pinned_memory_vector " << hexpointer(m_array_) << hexpointer(m_region_) << " resizing from " << m_size_ << " to " << s);
        if (m_region_) {
            if (s>m_region_->getLength()) {
                LOG_ERROR_MSG("Resizing from " << m_region_->getLength() << " to " << s)
                std::terminate();
                throw std::runtime_error("pinned_memory_vector should never be resized once an allocation has been assigned");
            }
            m_size_ = s;
        }
        else {
            m_region_ = m_alloc_->allocateRegion(s);
            m_array_ = static_cast<T*>(m_region_->getAddress());
            m_size_ = s;
        }
    }
    
    void reserve(std::size_t s) {
        LOG_DEBUG_MSG("pinned_memory_vector " << hexpointer(m_array_) << hexpointer(m_region_) " reserving from " << m_size_ << " to " << s);
        if (m_array_ || m_region_) {
            std::terminate();
            throw std::runtime_error("pinned_memory_vector should never be resized once an allocation has been assigned");            
        }
        m_region_ = m_alloc_->allocateRegion(s);
        m_array_ = static_cast<T*>(m_region_->getAddress());
    }

  private:
    pinned_memory_vector(pinned_memory_vector<T> const & other);

  };
}}}

#endif
