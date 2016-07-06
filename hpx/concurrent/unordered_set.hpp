////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CONCURRENT_UNORDERED_SET)
#define HPX_CONCURRENT_UNORDERED_SET

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
//
#include <unordered_set>
#include <utility>

// A quick wrapper around an unordered_set with a mutex to ensure two
// threads don't simultaneously write or read during write.
// Warning: not thread safe to use iterators whilst others are changing the set
// and several other functions are unsafe.

namespace hpx { namespace concurrent
{
    template<
        class Key,
        class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>,
        class Allocator = std::allocator<Key>
    >
    class unordered_set
    {
    private:
        typedef hpx::lcos::local::spinlock                mutex_type;
        typedef std::lock_guard<mutex_type>               scoped_lock;
        typedef std::unique_lock<mutex_type>              unique_lock;

    private:
        typedef std::unordered_set<Key, Hash, KeyEqual, Allocator> base_set;
        base_set            set_;
        mutable mutex_type  mutex_;

    public:
        typedef typename base_set::key_type key_type;
        typedef typename base_set::value_type value_type;
        typedef typename base_set::size_type size_type;
        typedef typename base_set::difference_type difference_type;
        typedef typename base_set::hasher hasher;
        typedef typename base_set::key_equal key_equal;
        typedef typename base_set::allocator_type allocator_type;
        typedef typename base_set::reference reference;
        typedef typename base_set::const_reference const_reference;
        typedef typename base_set::pointer pointer;
        typedef typename base_set::const_pointer const_pointer;
        typedef typename base_set::iterator iterator;
        typedef typename base_set::const_iterator const_iterator;
        typedef typename base_set::local_iterator local_iterator;
        typedef typename base_set::const_local_iterator const_local_iterator;

    public:
        // construct/destroy/copy
        explicit unordered_set(size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : set_(n, hf, eql, a) {};

        template <typename InputIterator>
        unordered_set(InputIterator first, InputIterator last,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
            : set_(first, last, n, hf, eql, a) {};

        unordered_set(const unordered_set& other)
        : set_(other) {};

        unordered_set(const allocator_type& a)
        : set_(a) {};

        unordered_set(const unordered_set& other, const allocator_type& a)
        : set_(other, a) {};

        //C++11 specific
        unordered_set(unordered_set&& other)
        : set_(std::forward<unordered_set>(other)) {};

        unordered_set(unordered_set&& other, const allocator_type& a)
        : set_(std::forward<unordered_set>(other), a) {};

        unordered_set(std::initializer_list<value_type> il,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : set_(il, n, hf, eql, a) {};

        ~unordered_set() {};

        unordered_set& operator=(const unordered_set& other)
        { return set_=other; }

        //C++11 specific
        unordered_set& operator=(unordered_set&& other)
        { return set_=other; }

        unordered_set& operator=(std::initializer_list<value_type> il)
        { return set_=il; }

        void swap(unordered_set& other)
        { set_.swap(other); }

        //
        // modifiers
        //
        std::pair<iterator, bool> insert(const value_type& x)
        {
            scoped_lock lock(mutex_);
            return set_.insert(x);
        }

        iterator insert(const_iterator hint, const value_type& x)
        {
            scoped_lock lock(mutex_);
            return set_.insert(hint, x);
        }

        template<class InputIterator>
        void insert(InputIterator first,InputIterator last)
        {
            scoped_lock lock(mutex_);
            set_.insert(first, last);
        }

        // C++11 specific
        std::pair<iterator, bool> insert(value_type&& x)
         {
            scoped_lock lock(mutex_);
            return set_.insert(std::forward<value_type>(x));
         }

        iterator insert(const_iterator hint, value_type&& x)
        {
            scoped_lock lock(mutex_);
            return set_.insert(hint, std::forward<value_type>(x));
        }

        void insert(std::initializer_list<value_type> il)
        {
            scoped_lock lock(mutex_);
            set_.insert(il);
        }

        // C++11 specific
        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            scoped_lock lock(mutex_);
            set_.emplace(std::forward<Args>(args)...);
        }

        template<typename... Args>
        iterator emplace_hint(const_iterator hint, Args&&... args)
        {
            scoped_lock lock(mutex_);
            return set_.emplace_hint(hint, std::forward<Args>(args)...);
        }

        // unsafe modifiers
        iterator unsafe_erase(const_iterator position)
        {
            scoped_lock lock(mutex_);
            return set_.unsafe_erase(position);
        }

        size_type unsafe_erase(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return set_.unsafe_erase(k);
        }

        iterator unsafe_erase(const_iterator first, const_iterator last)
        {
            scoped_lock lock(mutex_);
            return set_.unsafe_erase(first, last);
        }

        void clear()
        {
            scoped_lock lock(mutex_);
            set_.clear();
        }

        //
        // size and capacity
        //
        bool empty() const
        {
            scoped_lock lock(mutex_);
            return set_.empty();
        }
        size_type size() const
        {
            scoped_lock lock(mutex_);
            return set_.size();
        }
        size_type max_size() const
        {
            scoped_lock lock(mutex_);
            return set_.max_size();
        };

        //
        // iterators - not thread safe to access these
        //
        iterator begin() { return set_.begin(); }
        const_iterator begin() const  { return set_.begin(); };
        iterator end()  { return set_.end(); };
        const_iterator end() const  { return set_.end(); };
        const_iterator cbegin() const  { return set_.cbegin(); };
        const_iterator cend() const  { return set_.cbegin(); };

        //
        // observers
        //
        hasher hash_function() const { return set_.hash_function(); }
        key_equal key_eq() const { return set_.key_eq(); }

        //
        // lookup
        //
        iterator find(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return set_.find(k);
        }

        const_iterator find(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return set_.find(k);
        }

        size_type count(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return set_.count(k);
        }

        std::pair<iterator, iterator> equal_range(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return set_.equal_range(k);
        }

        std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return set_.equal_range(k);
        };

    };

#endif
}}
