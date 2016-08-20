//  Copyright (c) 2016 John Biddiscombe
//  Copyright (c) 2012 Scott Downie, Tag Games Limited
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
//  vector.h
//  Chilli Source
//  Created by Scott Downie on 19/09/2015.
//
//  The MIT License (MIT)
//
//  Copyright (c) 2012 Tag Games Limited
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//  THE SOFTWARE.
//

#ifndef _HPX_CONCURRENTVECTOR_H_
#define _HPX_CONCURRENTVECTOR_H_

#include <hpx/config.hpp>
#include <hpx/lcos/local/recursive_mutex.hpp>

#include <hpx/concurrent/vector_forward_iterator.hpp>
#include <hpx/concurrent/vector_reverse_iterator.hpp>
#include <hpx/concurrent/vector_const_forward_iterator.hpp>
#include <hpx/concurrent/vector_const_reverse_iterator.hpp>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <vector>

namespace hpx { namespace concurrent
{
    //------------------------------------------------------------------------
    /// Concurrent vector is a thread safe dynamic array implementation.
    /// It preserves the integrity of the array when accessing from different
    /// threads and also from changes to the array while iterating
    ///
    /// NOTE: This class syntax mimics STL and therefore does not use the CS
    ///       coding standards.
    ///
    /// @author S Downie
    //------------------------------------------------------------------------
    template <typename TType, typename Allocator = std::allocator<TType> >
    class vector
    {
        typedef std::int_least32_t s32;
        typedef hpx::lcos::local::recursive_mutex recursive_mutex;

        typedef typename std::allocator_traits<Allocator>::
                template rebind_alloc<std::pair<TType, bool> >
            allocator_type;

    public:
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator = vector_forward_iterator<TType, Allocator>;
        using const_iterator = vector_const_forward_iterator<TType, Allocator>;
        using reverse_iterator = vector_reverse_iterator<TType, Allocator>;
        using const_reverse_iterator =
            vector_const_reverse_iterator<TType, Allocator>;

        //--------------------------------------------------------------------
        /// Constructor default
        ///
        /// @author S Downie
        //--------------------------------------------------------------------
        vector();

        vector(size_type count, TType const& value = TType(),
            Allocator const& alloc = Allocator());

        //--------------------------------------------------------------------
        /// Construct from initialiser list
        ///
        /// @author S Downie
        ///
        /// @param Initialiser list
        //--------------------------------------------------------------------
        vector(std::initializer_list<TType>&& in_initialObjects);
        //--------------------------------------------------------------------
        /// Copy constructor that creates this as a copy of the given vector
        ///
        /// @author S Downie
        ///
        /// @param Vector to copy
        //--------------------------------------------------------------------
        vector(const vector& rhs);
        //--------------------------------------------------------------------
        /// Copy assignment that creates this as a copy of the given vector
        ///
        /// @author S Downie
        ///
        /// @param Vector to copy
        ///
        /// @return This as a copy
        //--------------------------------------------------------------------
        vector& operator=(const vector& rhs);
        //--------------------------------------------------------------------
        /// Move constructor that transfers ownership from the given vector
        ///
        /// @author S Downie
        ///
        /// @param Vector to move
        //--------------------------------------------------------------------
        vector(vector&& rhs);
        //--------------------------------------------------------------------
        /// Move assignment that transfers ownership from the given vector
        ///
        /// @author S Downie
        ///
        /// @param Vector to move
        ///
        /// @return This having taken ownership of the given vector
        //--------------------------------------------------------------------
        vector& operator=(vector&& rhs);
        //--------------------------------------------------------------------
        /// Push the object onto the back of the array. Unlike STL this does
        /// not invalidate any iterators
        ///
        /// @author S Downie
        ///
        /// @param Object to add
        //--------------------------------------------------------------------
        void push_back(TType&& in_object);
        //--------------------------------------------------------------------
        /// Push the object onto the back of the array. Unlike STL this does
        /// not invalidate any iterators
        ///
        /// @author S Downie
        ///
        /// @param Object to add
        //--------------------------------------------------------------------
        void push_back(const TType& in_object);
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return The object in the first element of the array (undefined if
        ///         empty)
        //--------------------------------------------------------------------
        TType& front();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return The object in the first element of the array (undefined if
        ///         empty)
        //--------------------------------------------------------------------
        const TType& front() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return The object in the last element of the array (undefined if
        ///         empty)
        //--------------------------------------------------------------------
        TType& back();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return The object in the last element of the array (undefined if
        ///         empty)
        //--------------------------------------------------------------------
        const TType& back() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @param Index
        ///
        /// @return The object at the given index of the array (undefined if
        ///         out of bounds)
        //--------------------------------------------------------------------
        TType& at(size_type in_index);
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @param Index
        ///
        /// @return The object at the given index of the array (undefined if
        ///         out of bounds)
        //--------------------------------------------------------------------
        const TType& at(size_type in_index) const;
        //--------------------------------------------------------------------
        /// Lock the array which prevents its contents being modified unsafely.
        ///
        /// @author S Downie
        //--------------------------------------------------------------------
        void lock();
        //--------------------------------------------------------------------
        /// Unlock the array.
        ///
        /// @author S Downie
        //--------------------------------------------------------------------
        void unlock();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return The number of items currently in the vector
        //--------------------------------------------------------------------
        size_type size() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Whether the vector is empty or not
        //--------------------------------------------------------------------
        bool empty() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector
        //--------------------------------------------------------------------
        iterator begin();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the end of the vector (the end being)
        /// after the last element
        //--------------------------------------------------------------------
        iterator end();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector
        //--------------------------------------------------------------------
        const_iterator begin() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the end of the vector (the end being)
        /// after the last element
        //--------------------------------------------------------------------
        const_iterator end() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector
        //--------------------------------------------------------------------
        const_iterator cbegin() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the end of the vector (the end being)
        /// after the last element
        //--------------------------------------------------------------------
        const_iterator cend() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the last element of the vector
        //--------------------------------------------------------------------
        reverse_iterator rbegin();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector (the
        ///         beginning in this case being before the first element)
        //--------------------------------------------------------------------
        reverse_iterator rend();
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the last element of the vector
        //--------------------------------------------------------------------
        const_reverse_iterator rbegin() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector (the
        ///         beginning in this case being before the first element)
        //--------------------------------------------------------------------
        const_reverse_iterator rend() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the last element of the vector
        //--------------------------------------------------------------------
        const_reverse_iterator crbegin() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @return Iterator pointing to the beginning of the vector (the
        ///         beginningin this case being before the first element)
        //--------------------------------------------------------------------
        const_reverse_iterator crend() const;
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @param Index
        ///
        /// @return The object at the given index of the array (undefined if
        ///         out of bounds)
        //--------------------------------------------------------------------
        TType& operator[](size_type in_index);
        //--------------------------------------------------------------------
        /// @author S Downie
        ///
        /// @param Index
        ///
        /// @return The object at the given index of the array (undefined if
        ///         out of bounds)
        //--------------------------------------------------------------------
        const TType& operator[](size_type in_index) const;
        //--------------------------------------------------------------------
        /// Remove the object from the vector that is pointed to by the given
        /// iterator. Unlike STL this does not invalidate any iterators
        ///
        /// @author S Downie
        ///
        /// @param Iterator
        ///
        /// @return The next iterator.
        //--------------------------------------------------------------------
        iterator erase(const iterator& in_it_erase);
        //--------------------------------------------------------------------
        /// Remove the object from the vector that is pointed to by the given
        /// iterator. Unlike STL this does not invalidate any iterators
        ///
        /// @author S Downie
        ///
        /// @param Iterator
        ///
        /// @return The next iterator.
        //--------------------------------------------------------------------
        reverse_iterator erase(const reverse_iterator& in_it_erase);
        //--------------------------------------------------------------------
        /// Clears the vector. Unlike STL this does not invalidate any iterators
        ///
        /// @author S Downie
        //--------------------------------------------------------------------
        void clear();

    private:
        //--------------------------------------------------------------------
        /// Cleanup any elements that are marked for removal
        ///
        /// @author S Downie
        //--------------------------------------------------------------------
        void garbage_collect();

    private:
        std::vector<std::pair<TType, bool>, allocator_type> container_;

        std::atomic<size_type> size_;
        bool is_locked_ = false;
        bool requires_gc_ = false;

        mutable recursive_mutex mutex_;
        std::unique_lock<recursive_mutex> lock_;
        s32 lock_count_ = 0;
    };

    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>::vector()
      : size_(0)
      , lock_(mutex_, std::defer_lock)
    {
    }
    //--------------------------------------------------------------------

    template <typename TType, typename Allocator>
    vector<TType, Allocator>::vector(size_type count, TType const& value,
            Allocator const& alloc)
      : container_(count, std::make_pair(value, false), allocator_type(alloc))
      , size_(count)
      , lock_(mutex_, std::defer_lock)
    {
    }

    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>::vector(
        std::initializer_list<TType>&& in_initial_objects)
      : size_(0)
      , lock_(mutex_, std::defer_lock)
    {
        size_ = in_initial_objects.size();
        container_.reserve(size_);

        for (const auto& object : in_initial_objects)
        {
            container_.push_back(std::make_pair(object, false));
        }
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>::vector(const vector& rhs)
      : container_(rhs.container_)
      , size_(rhs.size_.load())
      , lock_(mutex_, std::defer_lock)
    {
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>& vector<TType, Allocator>::operator=(
        const vector<TType, Allocator>& rhs)
    {
        size_ = rhs.size_.load();
        container_ = rhs.container_;
        return *this;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>::vector(vector&& rhs)
      : container_(rhs.container_)
      , size_(rhs.size_.load())
      , lock_(mutex_, std::defer_lock)
    {
        rhs.size_ = 0;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector<TType, Allocator>& vector<TType, Allocator>::operator=(
        vector<TType, Allocator>&& rhs)
    {
        size_ = rhs.size_.load();
        rhs.size_ = 0;
        container_ = std::move(rhs.container_);
        return *this;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::push_back(TType&& in_object)
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        container_.push_back(
            std::make_pair(std::forward<TType>(in_object), false));
        ++size_;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::push_back(const TType& in_object)
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        container_.push_back(std::make_pair(in_object, false));
        ++size_;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    TType& vector<TType, Allocator>::front()
    {
        return at(0);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    const TType& vector<TType, Allocator>::front() const
    {
        return at(0);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    TType& vector<TType, Allocator>::back()
    {
        return at(size() - 1);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    const TType& vector<TType, Allocator>::back() const
    {
        return at(size() - 1);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    TType& vector<TType, Allocator>::at(size_type in_index)
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        if (!requires_gc_)
        {
            return container_.at(in_index).first;
        }
        else
        {
            size_type size = container_.size();
            difference_type count = -1;
            size_type index = 0;
            for (size_type i = 0; i < size; ++i)
            {
                if (container_[i].second == false)
                {
                    if (++count == in_index)
                    {
                        index = i;
                        break;
                    }
                }
            }

            return container_[index].first;
        }
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    const TType& vector<TType, Allocator>::at(size_type in_index) const
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        if (!requires_gc_)
        {
            return container_.at(in_index).first;
        }
        else
        {
            size_type size = container_.size();
            difference_type count = -1;
            size_type index = 0;
            for (size_type i = 0; i < size; ++i)
            {
                if (container_[i].second == false)
                {
                    if (++count == in_index)
                    {
                        index = i;
                        break;
                    }
                }
            }

            return container_[index].first;
        }
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::lock()
    {
        std::unique_lock<recursive_mutex> l(mutex_);

        if (lock_count_ == 0)
        {
            lock_.lock();
        }

        lock_count_++;
        is_locked_ = true;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::unlock()
    {
        std::unique_lock<recursive_mutex> l(mutex_);

        is_locked_ = false;
        lock_count_--;

        if (lock_count_ == 0)
        {
            garbage_collect();
            lock_.unlock();
        }
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::size_type
    vector<TType, Allocator>::size() const
    {
        return size_;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    bool vector<TType, Allocator>::empty() const
    {
        return size_ == 0;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::iterator
    vector<TType, Allocator>::begin()
    {
        return iterator(&container_, &mutex_);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::iterator vector<TType, Allocator>::end()
    {
        return iterator(&container_, &mutex_, size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_iterator
    vector<TType, Allocator>::begin() const
    {
        return const_iterator(
            &container_, const_cast<recursive_mutex*>(&mutex_));
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_iterator
    vector<TType, Allocator>::end() const
    {
        return const_iterator(&container_,
            const_cast<recursive_mutex*>(&mutex_), size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_iterator
    vector<TType, Allocator>::cbegin() const
    {
        return const_iterator(
            &container_, const_cast<recursive_mutex*>(&mutex_));
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_iterator
    vector<TType, Allocator>::cend() const
    {
        return const_iterator(&container_,
            const_cast<recursive_mutex*>(&mutex_), size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::reverse_iterator
    vector<TType, Allocator>::rbegin()
    {
        return reverse_iterator(&container_, &mutex_);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::reverse_iterator
    vector<TType, Allocator>::rend()
    {
        return reverse_iterator(&container_, &mutex_, size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_reverse_iterator
    vector<TType, Allocator>::rbegin() const
    {
        return const_reverse_iterator(
            &container_, const_cast<recursive_mutex*>(&mutex_));
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_reverse_iterator
    vector<TType, Allocator>::rend() const
    {
        return const_reverse_iterator(&container_,
            const_cast<recursive_mutex*>(&mutex_), size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_reverse_iterator
    vector<TType, Allocator>::crbegin() const
    {
        return const_reverse_iterator(
            &container_, const_cast<recursive_mutex*>(&mutex_));
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    typename vector<TType, Allocator>::const_reverse_iterator
    vector<TType, Allocator>::crend() const
    {
        return const_reverse_iterator(&container_,
            const_cast<recursive_mutex*>(&mutex_), size());
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    TType& vector<TType, Allocator>::operator[](size_type in_index)
    {
        return at(in_index);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    const TType& vector<TType, Allocator>::operator[](size_type in_index) const
    {
        return at(in_index);
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector_forward_iterator<TType, Allocator>
    vector<TType, Allocator>::erase(
        const vector_forward_iterator<TType, Allocator>& in_it_erase)
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        if (is_locked_ == false)
        {
            container_.erase(container_.begin() + in_it_erase.get_index());
        }
        else
        {
            container_[in_it_erase.get_index()].second = true;
            requires_gc_ = true;
        }

        size_--;

        auto it_copy = in_it_erase;
        ++it_copy;
        return it_copy;
    }
    //--------------------------------------------------------------------
    /// This uses the
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    vector_reverse_iterator<TType, Allocator> vector<TType, Allocator>::erase(
        const vector_reverse_iterator<TType, Allocator>& in_it_erase)
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        if (is_locked_ == false)
        {
            container_.erase(container_.begin() + in_it_erase.get_index());
        }
        else
        {
            container_[in_it_erase.get_index()].second = true;
            requires_gc_ = true;
        }

        size_--;

        auto itCopy = in_it_erase;
        ++itCopy;
        return itCopy;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::clear()
    {
        std::unique_lock<recursive_mutex> l(mutex_);
        if (is_locked_ == false)
        {
            container_.clear();
        }
        else
        {
            for (auto& object : container_)
            {
                object.second = true;
            }

            requires_gc_ = true;
        }

        size_ = 0;
    }
    //--------------------------------------------------------------------
    //--------------------------------------------------------------------
    template <typename TType, typename Allocator>
    void vector<TType, Allocator>::garbage_collect()
    {
        for (auto it = container_.begin(); it != container_.end(); /**/)
        {
            if (it->second)
            {
                it = container_.erase(it);
            }
            else
            {
                ++it;
            }
        }

        requires_gc_ = false;
    }
}}

#endif
