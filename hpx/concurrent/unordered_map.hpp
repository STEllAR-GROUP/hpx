////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CONCURRENT_UNORDERED_MAP)
#define HPX_CONCURRENT_UNORDERED_MAP

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/spinlock.hpp>
//
#include <unordered_map>
#include <utility>

// A quick wrapper around an unordered_map with a mutex to ensure two
// threads don't simultaneously write or read during write.
// Warning: not thread safe to use iterators whilst others are changing the set
// and several other functions are unsafe.

namespace hpx { namespace concurrent
{
    template<
        class Key,
        class Value,
        class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>,
        class Allocator = std::allocator<std::pair<const Key, Value>>
    >
    class unordered_map
    {
    private:
        typedef hpx::lcos::local::spinlock                mutex_type;
        typedef std::lock_guard<mutex_type>               scoped_lock;
        typedef std::unique_lock<mutex_type>              unique_lock;

    private:
        typedef std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> base_map;
        base_map            map_;
        mutable mutex_type  mutex_;

    public:
        typedef typename base_map::key_type key_type;
        typedef typename base_map::mapped_type mapped_type;
        typedef typename base_map::value_type value_type;
        typedef typename base_map::size_type size_type;
        typedef typename base_map::difference_type difference_type;
        typedef typename base_map::hasher hasher;
        typedef typename base_map::key_equal key_equal;
        typedef typename base_map::allocator_type allocator_type;
        typedef typename base_map::reference reference;
        typedef typename base_map::const_reference const_reference;
        typedef typename base_map::pointer pointer;
        typedef typename base_map::const_pointer const_pointer;
        typedef typename base_map::iterator iterator;
        typedef typename base_map::const_iterator const_iterator;
        typedef typename base_map::local_iterator local_iterator;
        typedef typename base_map::const_local_iterator const_local_iterator;

    public:
        //
        // construct/destroy/copy
        //
        explicit unordered_map(size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : map_(n, hf, eql, a) {};

        template <typename InputIterator>
        unordered_map(InputIterator first, InputIterator last,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
            : map_(first, last, n, hf, eql, a) {};

        unordered_map(const unordered_map& other)
        : map_(other) {};

        unordered_map(const allocator_type& a)
        : map_(a) {};

        unordered_map(const unordered_map& other, const allocator_type& a)
        : map_(other, a) {};

        //C++11 specific
        unordered_map(unordered_map&& other)
        : map_(std::forward<unordered_map>(other)) {};

        unordered_map(unordered_map&& other, const allocator_type& a)
        : map_(std::forward<unordered_map>(other), a) {};

        unordered_map(std::initializer_list<value_type> il,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : map_(il, n, hf, eql, a) {};

        ~unordered_map() {};

        unordered_map& operator=(const unordered_map& other)
        { return map_=other; }

        //C++11 specific
        unordered_map& operator=(unordered_map&& other)
        { return map_=other; }

        unordered_map& operator=(std::initializer_list<value_type> il)
        { return map_=il; }

        void swap(unordered_map& other)
        { map_.swap(other); }

        //
        // modifiers
        //
        std::pair<iterator, bool> insert(const value_type& x)
        {
            scoped_lock lock(mutex_);
            return map_.insert(x);
        }

        iterator insert(const_iterator hint, const value_type& x)
        {
            scoped_lock lock(mutex_);
            return map_.insert(hint, x);
        }

        template<class InputIterator>
        void insert(InputIterator first,InputIterator last)
        {
            scoped_lock lock(mutex_);
            map_.insert(first, last);
        }

        // C++11 specific
        std::pair<iterator, bool> insert(value_type&& x)
         {
            scoped_lock lock(mutex_);
            return map_.insert(std::forward<value_type>(x));
         }

        iterator insert(const_iterator hint, value_type&& x)
        {
            scoped_lock lock(mutex_);
            return map_.insert(hint, std::forward<value_type>(x));
        }

        void insert(std::initializer_list<value_type> il)
        {
            scoped_lock lock(mutex_);
            map_.insert(il);
        }

        // C++11 specific
        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            scoped_lock lock(mutex_);
            map_.emplace(std::forward<Args>(args)...);
        }

        template<typename... Args>
        iterator emplace_hint(const_iterator hint, Args&&... args)
        {
            scoped_lock lock(mutex_);
            return map_.emplace_hint(hint, std::forward<Args>(args)...);
        }

        // unsafe modifiers
        iterator unsafe_erase(const_iterator position)
        {
            scoped_lock lock(mutex_);
            return map_.unsafe_erase(position);
        }

        size_type unsafe_erase(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return map_.unsafe_erase(k);
        }

        iterator unsafe_erase(const_iterator first, const_iterator last)
        {
            scoped_lock lock(mutex_);
            return map_.unsafe_erase(first, last);
        }

        void clear()
        {
            scoped_lock lock(mutex_);
            map_.clear();
        }

        //
        // size and capacity
        //
        bool empty() const
        {
            scoped_lock lock(mutex_);
            return map_.empty();
        }
        size_type size() const
        {
            scoped_lock lock(mutex_);
            return map_.size();
        }
        size_type max_size() const
        {
            scoped_lock lock(mutex_);
            return map_.max_size();
        };

        //
        // iterators - not thread safe to access these
        //
        iterator begin() { return map_.begin(); }
        const_iterator begin() const  { return map_.begin(); };
        iterator end()  { return map_.end(); };
        const_iterator end() const  { return map_.end(); };
        const_iterator cbegin() const  { return map_.cbegin(); };
        const_iterator cend() const  { return map_.cbegin(); };

        //
        // observers
        //
        hasher hash_function() const { return map_.hash_function(); }
        key_equal key_eq() const { return map_.key_eq(); }

        //
        // lookup
        //
        iterator find(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return map_.find(k);
        }

        const_iterator find(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return map_.find(k);
        }

        size_type count(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return map_.count(k);
        }

        std::pair<iterator, iterator> equal_range(const key_type& k)
        {
            scoped_lock lock(mutex_);
            return map_.equal_range(k);
        }

        std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return map_.equal_range(k);
        };

        //
        // map operators
        //
        mapped_type& operator[](const key_type& k)
        {
            scoped_lock lock(mutex_);
            return map_[k];
        };

        mapped_type& at( const key_type& k )
        {
            scoped_lock lock(mutex_);
            return map_.at(k);
        };

        const mapped_type& at(const key_type& k) const
        {
            scoped_lock lock(mutex_);
            return map_.at(k);
        };

    };

#endif
}}