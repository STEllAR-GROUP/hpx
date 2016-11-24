////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 John Biddiscombe
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_CONCURRENT_UNORDERED_MAP)
#define HPX_CONCURRENT_UNORDERED_MAP

#include <hpx/lcos/local/mutex.hpp>
#include <hpx/lcos/local/shared_mutex.hpp>
#include <plugins/parcelport/verbs/readers_writers_mutex.hpp>
#include <plugins/parcelport/verbs/rdma/rdma_locks.hpp>
//
#include <unordered_map>
#include <utility>
#include <shared_mutex>

// A quick wrapper around an unordered_map with a mutex to ensure two
// threads don't simultaneously write or read during write.
// Warning: not thread safe to use iterators whilst others are changing the map
// obtain a read lock before iterating using the provided mutex and function
//   map_type::map_read_lock_type read_lock(map.read_write_mutex());
// in order to safely iterate over contents and block any writers from gaining access
//
namespace hpx {
namespace concurrent {

    template<
        class Key,
        class Value,
        class Hash = std::hash<Key>,
        class KeyEqual = std::equal_to<Key>,
        class Allocator = std::allocator<std::pair<const Key, Value>>
    >
    class unordered_map
    {
    public:

        typedef hpx::lcos::local::readers_writer_mutex                      rw_mutex_type;
        typedef hpx::parcelset::policies::verbs::unique_lock<rw_mutex_type> write_lock;
        typedef boost::shared_lock<rw_mutex_type>                           read_lock;
        typedef boost::defer_lock_t                                         defer_lock;

    private:
        typedef std::unordered_map<Key, Value, Hash, KeyEqual, Allocator> base_map;
        base_map              map_;
        mutable rw_mutex_type mutex_;
        mutable read_lock     iterator_lock_;

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
        //
        typedef read_lock  map_read_lock_type;
        typedef write_lock map_write_lock_type;

    public:
        //
        // construct/destroy/copy
        //
        explicit unordered_map(size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : map_(n, hf, eql, a)
        , iterator_lock_(mutex_, defer_lock())
        {}

        template <typename InputIterator>
        unordered_map(InputIterator first, InputIterator last,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
            : map_(first, last, n, hf, eql, a)
            , iterator_lock_(mutex_, defer_lock())
            {}

        unordered_map(const unordered_map& other)
        : map_(other)
        , iterator_lock_(mutex_, defer_lock())
        {}

        unordered_map(const allocator_type& a)
        : map_(a)
        , iterator_lock_(mutex_, defer_lock())
        {}

        unordered_map(const unordered_map& other, const allocator_type& a)
        : map_(other, a)
        , iterator_lock_(mutex_, defer_lock())
        {}

        // C++11 specific
        unordered_map(unordered_map&& other)
        : map_(std::forward<unordered_map>(other))
        , iterator_lock_(mutex_, defer_lock())
        {}

        unordered_map(unordered_map&& other, const allocator_type& a)
        : map_(std::forward<unordered_map>(other), a)
        , iterator_lock_(mutex_, defer_lock())
        {}

        unordered_map(std::initializer_list<value_type> il,
            size_type n = 64,
            const hasher& hf = hasher(),
            const key_equal& eql = key_equal(),
            const allocator_type& a = allocator_type())
        : map_(il, n, hf, eql, a)
        , iterator_lock_(mutex_, defer_lock())
        {}

        ~unordered_map() {};

        unordered_map& operator=(const unordered_map& other)
        {
            write_lock lock(mutex_);
            return map_ = other;
        }

        // C++11 specific
        unordered_map& operator=(unordered_map&& other)
        {
            write_lock lock(mutex_);
            return map_ = std::forward<unordered_map>(other);
        }

        unordered_map& operator=(std::initializer_list<value_type> il)
        {
            write_lock lock(mutex_);
            return map_ = il;
        }

        void swap(unordered_map& other)
        {
            write_lock lock(mutex_);
            map_.swap(other);
        }

        //
        // modifiers
        //
        std::pair<iterator, bool> insert(const value_type& x)
        {
            write_lock lock(mutex_);
            return map_.insert(x);
        }

        iterator insert(const_iterator hint, const value_type& x)
        {
            write_lock lock(mutex_);
            return map_.insert(hint, x);
        }

        template<class InputIterator>
        void insert(InputIterator first, InputIterator last)
        {
            write_lock lock(mutex_);
            map_.insert(first, last);
        }

        // C++11 specific
        std::pair<iterator, bool> insert(value_type&& x)
         {
            write_lock lock(mutex_);
            return map_.insert(std::forward<value_type>(x));
         }

        iterator insert(const_iterator hint, value_type&& x)
        {
            write_lock lock(mutex_);
            return map_.insert(hint, std::forward<value_type>(x));
        }

        void insert(std::initializer_list<value_type> il)
        {
            write_lock lock(mutex_);
            map_.insert(il);
        }

        // C++11 specific
        template<typename... Args>
        std::pair<iterator, bool> emplace(Args&&... args)
        {
            write_lock lock(mutex_);
            map_.emplace(std::forward<Args>(args)...);
        }

        template<typename... Args>
        iterator emplace_hint(const_iterator hint, Args&&... args)
        {
            write_lock lock(mutex_);
            return map_.emplace_hint(hint, std::forward<Args>(args)...);
        }

        // modifiers
        iterator erase(const_iterator position)
        {
            write_lock lock(mutex_);
            return map_.erase(position);
        }

        size_type erase(const key_type& k)
        {
            write_lock lock(mutex_);
            return map_.erase(k);
        }

        iterator erase(const_iterator first, const_iterator last)
        {
            write_lock lock(mutex_);
            return map_.erase(first, last);
        }

        void clear()
        {
            write_lock lock(mutex_);
            map_.clear();
        }

        //
        // size and capacity
        //
        bool empty() const
        {
            read_lock lock(mutex_);
            return map_.empty();
        }
        size_type size() const
        {
            read_lock lock(mutex_);
            return map_.size();
        }
        size_type max_size() const
        {
            read_lock lock(mutex_);
            return map_.max_size();
        };

        //
        // iterators - not thread safe to access these
        // obtain a read_lock before iterating, and release when done
        //
        iterator begin() { return map_.begin(); }
        const_iterator begin() const  { return map_.begin(); };
        iterator end()  { return map_.end(); };
        const_iterator end() const  { return map_.end(); };
        const_iterator cbegin() const  { return map_.cbegin(); };
        const_iterator cend() const  { return map_.cbegin(); };

        //
        // Before iterating over the map one must obtain a read lock,
        // one may use this mutex to gain a lock as follows
        //
        // map_type::map_read_lock_type read_lock(map.read_write_mutex());
        //
        rw_mutex_type& read_write_mutex() {
            return mutex_;
        }

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
            read_lock lock(mutex_);
            return map_.find(k);
        }

        const_iterator find(const key_type& k) const
        {
            read_lock lock(mutex_);
            return map_.find(k);
        }

        std::pair<const_iterator, bool> is_in_map(const key_type& k) const
        {
            read_lock lock(mutex_);
            const_iterator it = map_.find(k);
            bool result = (it != map_.end());
            return std::make_pair(it, result);;
        }

        size_type count(const key_type& k) const
        {
            read_lock lock(mutex_);
            return map_.count(k);
        }

        std::pair<iterator, iterator> equal_range(const key_type& k)
        {
            read_lock lock(mutex_);
            return map_.equal_range(k);
        }

        std::pair<const_iterator, const_iterator> equal_range(const key_type& k) const
        {
            read_lock lock(mutex_);
            return map_.equal_range(k);
        };

        //
        // map operators
        //

        //
        // operator[] should only be used for reading, if writing, use insert
        // which will take a write_lock for safety
        //
        const mapped_type& operator[](const key_type& k) const
        {
            read_lock lock(mutex_);
            return map_.at(k);
        };

        mapped_type& at( const key_type& k )
        {
            read_lock lock(mutex_);
            return map_.at(k);
        };

        const mapped_type& at(const key_type& k) const
        {
            read_lock lock(mutex_);
            return map_.at(k);
        };

    };

}}
#endif
