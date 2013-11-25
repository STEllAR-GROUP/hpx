////////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2011 Bryce Adelstein-Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_0097965D_313C_40DB_ABD0_5EA9537DDF8F)
#define HPX_0097965D_313C_40DB_ABD0_5EA9537DDF8F

#include <hpx/config.hpp>
#include <functional>
#include <iostream>
#include <list>

#include <hpx/util/move.hpp>
#include <hpx/assert.hpp>
#include <boost/integer.hpp>
#include <boost/icl/interval_set.hpp>
#include <boost/intrusive/set.hpp>
#include <boost/noncopyable.hpp>
#include <boost/foreach.hpp>

namespace hpx { namespace util
{

template <
    typename Key
  , typename Data
>
struct mapping
  : boost::intrusive::set_base_hook<
        boost::intrusive::optimize_size<true>
    >
{
    typedef boost::icl::closed_interval<Key, std::less> key_type;
    typedef Data data_type;

    key_type key_;
    data_type data_;

    mapping()
      : key_()
      , data_()
    {}

    template <
        typename T
    >
    mapping(
        key_type const& k
      , BOOST_FWD_REF(T) d
        )
      : key_(k)
      , data_(boost::forward<T>(d))
    {}

    template <
        typename T
    >
    mapping(
        Key const& lower
      , Key const& upper
      , BOOST_FWD_REF(T) d
        )
      : key_(lower, upper)
      , data_(boost::forward<T>(d))
    {}

    friend bool operator<(
        mapping const& a
      , mapping const& b
        )
    {
        return boost::icl::exclusive_less(a.key_, b.key_);
    }

    template <
        typename T
    >
    friend bool operator<(
        T const& a
      , mapping const& b
        )
    {
        key_type const a_(a);
        return boost::icl::exclusive_less(a_, b.key_);
    }

    template <
        typename T
    >
    friend bool operator<(
        mapping const& a
      , T const& b
        )
    {
        key_type const b_(b);
        return boost::icl::exclusive_less(a.key_, b_);
    }

    friend bool operator<(
        key_type const& a
      , mapping const& b
        )
    {
        return boost::icl::exclusive_less(a, b.key_);
    }

    friend bool operator<(
        mapping const& a
      , key_type const& b
        )
    {
        return boost::icl::exclusive_less(a.key_, b);
    }

    key_type const& key() const
    {
        return key_;
    }

    data_type const& data() const
    {
        return data_;
    }
};

template <
    typename Key
>
boost::icl::closed_interval<Key, std::less> partition(
    Key const& base
  , Key const& size
    )
{
    Key const upper = size ? (base + size - 1) : 0;
    return boost::icl::closed_interval<Key, std::less>(base, upper);
}

template <
    typename Key
  , typename Data
>
boost::icl::closed_interval<Key, std::less> point(
    Key const& key
    )
{
    return boost::icl::closed_interval<Key, std::less>(key, key);
}

struct polymorphic_less
{
    template <
        typename T0
      , typename T1
    >
    bool operator()(
        T0 const& a
      , T1 const& b
        ) const
    {
        return a < b;
    }
};

template <
    typename T
>
bool mergeable(
    T const& left
  , T const& right
    )
{
    return 0 == boost::icl::distance(left, right);
}

// TODO: Check for mergability before allocating nodes.
template <
    typename Key
  , typename Data
>
struct merging_map : boost::noncopyable
{
    typedef mapping<Key, Data> value_type;
    typedef typename value_type::key_type key_type;
    typedef typename value_type::data_type data_type;

    typedef boost::intrusive::set<
        value_type
      , boost::intrusive::constant_time_size<true>
      , boost::intrusive::compare<polymorphic_less>
    > map_type;

    typedef typename map_type::iterator iterator;
    typedef typename map_type::const_iterator const_iterator;
    typedef typename map_type::reference reference;
    typedef typename map_type::const_reference const_reference;
    typedef typename map_type::pointer pointer;
    typedef typename map_type::difference_type difference_type;
    typedef typename map_type::size_type size_type;

    struct disposer
    {
        void operator()(
            pointer p
            ) const
        {
            if (p)
            {
                delete p;
                p = 0;
            }
        }
    };

  private:
    map_type map_;

    ///////////////////////////////////////////////////////////////////////////
    /// Insert the key-value pair formed by \p key and \p data before \p pos.
    template <
        typename T
    >
    iterator insert_before(
        const_iterator pos
      , key_type const& key
      , BOOST_FWD_REF(T) data
        )
    {
        value_type* node = new value_type(key, boost::forward<T>(data));
        return map_.insert_before(pos, *node);
    }

    /// Insert \p node into the map before \p pos.
    iterator insert_before(
        const_iterator pos
      , value_type* node
        )
    {
        HPX_ASSERT(node);
        return map_.insert_before(pos, *node);
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Attempt to merge \p pos with the next lowest node.
    iterator lower_merge(
        iterator pos
        )
    { // {{{
        if (pos == end() || pos == begin())
            return pos;

        iterator next = pos; --next;

        // Does the next lowest node touch pos?
        if (mergeable(pos->key_, next->key_))
        {
            if (pos->data_ == next->data_)
            {
                // Erase the lower of the two mappings.
                key_type const key = next->key_;
                map_.erase_and_dispose(next, disposer());

                // Extend the higher mapping to cover the lower mapping.
                pos->key_ = boost::icl::hull(key, pos->key_);

                return pos;
            }
        }

        return pos;
    } // }}}

    /// Attempt to merge \p pos with the next highest node. Might invalidate
    /// \p pos.
    iterator upper_merge(
        iterator pos
        )
    { // {{{
        if (pos == end())
            return pos;

        iterator next = pos; ++next;

        if (next == end())
            return pos;

        // Does the next highest node touch pos?
        if (mergeable(pos->key_, next->key_))
        {
            // Is the data equivalent?
            if (pos->data_ == next->data_)
            {
                // Erase the lower of the two mappings.
                key_type const key = pos->key_;
                map_.erase_and_dispose(pos, disposer());

                // Extend the higher mapping to cover the lower mapping.
                next->key_ = boost::icl::hull(key, next->key_);

                return next;
            }
        }

        return pos;
    } // }}}

    /// Attempt to merge \p pos with the next highest and next lowest nodes.
    /// Might invalidate \p pos.
    iterator merge(
        iterator pos
        )
    {
        return upper_merge(lower_merge(pos));
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Overwrite or split \p pos and insert a key-value pair formed by
    /// \p sub and \p data into the map. Merges any newly created or updated
    /// mappings if possible.
    template <
        typename T
    >
    iterator remap(
        iterator pos
      , key_type const& sub
      , BOOST_FWD_REF(T) data
        )
    { // {{{
        const_iterator cpos(pos);

        key_type const& sup = pos->key_;

        // super-object - [a, b]
        // sub-object   - [c, d]
        Key const a = boost::icl::lower(sup);
        Key const b = boost::icl::upper(sup);
        Key const c = boost::icl::lower(sub);
        Key const d = boost::icl::upper(sub);

        HPX_ASSERT(a <= c);
        HPX_ASSERT(d <= b);

        ///////////////////////////////////////////////////////////////////////
        // Check if we can merge the new mapping into the old one.
        if (pos->data_ == data)
        {
            // Okay, we're a no-op.
            return pos;
        }

        ///////////////////////////////////////////////////////////////////////
        // Update the existing mapping.
        // |a     b|
        // |c     d|
        else if (a == c && b == d)
        {
            pos->data_ = boost::forward<T>(data);

            // We've updated the existing mappings data, so we might now be
            // able to merge with one of our neighbors.
            return merge(pos);
        }

        key_type const db(d + 1, b); // (d, b]

        // Move the existing mapping.
        // |a     b|
        // |c d|
        if (a == c)
        {
            HPX_ASSERT(!boost::icl::is_empty(db));

            // Shrink the existing mapping.
            pos->key_ = db;

            // Insert the new mapping, and try to merge it with the next lowest
            // node. We don't try to do a merge with the next highest node,
            // because that's the existing mapping, and we know the data of the
            // existing and the new mappings are different if we've reached this
            // point in the code.
            return lower_merge(insert_before(cpos, sub, boost::forward<T>(data)));
        }

        // a < c, so c != 0
        HPX_ASSERT(c != 0);
        key_type const ac(a, c - 1); // [a, c)

        // Move the existing mapping.
        // |a     b|
        //     |c d|
        if (b == d)
        {
            HPX_ASSERT(!boost::icl::is_empty(ac));

            // Add a new node with an updated version of the existing mapping.
            // There is no point in trying to merge this new node, because its
            // value hasn't changed.
            insert_before(cpos, ac, pos->data_);

            // Replace the existing mapping with new mapping.
            pos->key_ = sub;
            pos->data_ = boost::forward<T>(data);

            // We don't try to do a merge with the next lowest node,
            // because that's the existing mapping, and we know the data of the
            // existing and the new mappings are different if we've reached this
            // point in the code.
            return upper_merge(pos);
        }

        // Split the existing mapping.
        // |a     b|
        //   |c d|
        else
        {
            HPX_ASSERT(!boost::icl::is_empty(ac));
            HPX_ASSERT(!boost::icl::is_empty(db));

            // Shrink the existing mapping.
            pos->key_ = db;

            // Add the new mapping before the existing mapping.
            iterator r = insert_before(cpos, sub, boost::forward<T>(data));

            // Add a modified copy of the existing mapping before the new
            // mapping.
            insert_before(const_iterator(r), ac, pos->data_);

            // The new mapping is bordered by the existing mapping, so no
            // merge is necessary.
            return r;
        }
    } // }}}

    /// Overwrite or split \p pos and insert \p node into the map. Merges any
    /// newly created or updated mappings if possible. Might delete \p node.
    iterator remap(
        iterator pos
      , value_type* node
        )
    { // {{{
        HPX_ASSERT(node);

        const_iterator cpos(pos);

        key_type const& sup = pos->key_;

        // super-object - [a, b]
        // sub-object   - [c, d]
        Key const a = boost::icl::lower(sup);
        Key const b = boost::icl::upper(sup);
        Key const c = boost::icl::lower(node->key_);
        Key const d = boost::icl::upper(node->key_);

        HPX_ASSERT(a <= c);
        HPX_ASSERT(d <= b);

        ///////////////////////////////////////////////////////////////////////
        // Check if we can merge the new mapping into the old one.
        if (pos->data_ == node->data_)
        {
            // Okay, we're a no-op, so we need to destroy the node.
            delete node;
            node = 0;

            return pos;
        }

        ///////////////////////////////////////////////////////////////////////
        // Update the existing mapping.
        // |a     b|
        // |c     d|
        else if (a == c && b == d)
        {
            pos->data_ = node->data_;

            delete node;
            node = 0;

            // We've updated the existing mappings data, so we might now be
            // able to merge with one of our neighbors.
            return merge(pos);
        }

        key_type const db(d + 1, b); // (d, b]

        // Move the existing mapping.
        // |a     b|
        // |c d|
        if (a == c)
        {
            HPX_ASSERT(!boost::icl::is_empty(db));

            // Shrink the existing mapping.
            pos->key_ = db;

            // Insert the new mapping, and try to merge it with the next lowest
            // node. We don't try to do a merge with the next highest node,
            // because that's the existing mapping, and we know the data of the
            // existing and the new mappings are different if we've reached this
            // point in the code.
            return lower_merge(insert_before(cpos, node));
        }

        // a < c, so c != 0
        HPX_ASSERT(c != 0);
        key_type const ac(a, c - 1); // [a, c)

        // Move the existing mapping.
        // |a     b|
        //     |c d|
        if (b == d)
        {
            HPX_ASSERT(!boost::icl::is_empty(ac));

            // Add a new node with an updated version of the existing mapping.
            // There is no point in trying to merge this new node, because its
            // value hasn't changed.
            insert_before(cpos, ac, pos->data_);

            // Replace the existing mapping with new mapping.
            *pos = *node;

            delete node;
            node = 0;

            // We don't try to do a merge with the next lowest node,
            // because that's the existing mapping, and we know the data of the
            // existing and the new mappings are different if we've reached this
            // point in the code.
            return upper_merge(pos);
        }

        // Split the existing mapping.
        // |a     b|
        //   |c d|
        else
        {
            HPX_ASSERT(!boost::icl::is_empty(ac));
            HPX_ASSERT(!boost::icl::is_empty(db));

            // Shrink the existing mapping.
            pos->key_ = db;

            // Add the new mapping before the existing mapping.
            iterator r = insert_before(cpos, node);

            // Add a modified copy of the existing mapping before the new
            // mapping.
            insert_before(const_iterator(r), ac, pos->data_);

            // The new mapping is bordered by the existing mapping, so no
            // merge is necessary.
            return r;
        }
    } // }}}

  public:
    merging_map() {}

    ~merging_map()
    {
        clear();
    }

    ///////////////////////////////////////////////////////////////////////////
    /// Bind the range formed by [\p lower, \p upper] to \p data, overwriting
    /// or splitting any existing mappings with the same key. Merges any newly
    /// created mappings if possible.
    template <
        typename T
    >
    iterator bind(
        Key const& lower
      , Key const& upper
      , BOOST_FWD_REF(T) data
        )
    {
        key_type const key(lower, upper);
        return bind(key, boost::forward<T>(data));
    }

    /// Bind \p key to \p data, remapping any existing mappings that overlap.
    /// Merges any newly created or updated mappings if possible.
    template <
        typename T
    >
    iterator bind(
        key_type const& key
      , BOOST_FWD_REF(T) data
        )
    { // {{{
        std::pair<iterator, iterator> matches = find(key);

        if (matches.first == end() && matches.second == end())
        {
            value_type* node = new value_type(key, boost::forward<T>(data));
            std::pair<iterator, bool> r = map_.insert(*node);
            HPX_ASSERT(r.second);
            return merge(r.first);
        }

        for (; matches.first != matches.second;)
        {
            key_type& match = matches.first->key_;

            // Is key a subset of match?
            if (boost::icl::contains(match, key))
                return remap(matches.first, key, boost::forward<T>(data));

            // Is match a subset of key?
            if (boost::icl::contains(key, match))
            {
                map_.erase_and_dispose(matches.first++, disposer());
                continue;
            }

            // match - [a, b]
            // key   - [c, d]
            Key const a = boost::icl::lower(match);
            Key const b = boost::icl::upper(match);
            Key const c = boost::icl::lower(key);
            Key const d = boost::icl::upper(key);

            // Does match begin before key?
            // |a  b|
            //   |c  d|
            if (a < c)
            {
                // a < c, so c != 0
                HPX_ASSERT(c != 0);

                // Shrink the existing mapping.
                match = key_type(a, c - 1); // [a, c)
            }

            // Does key end before match?
            //   |a  b|
            // |c  d|
            else if (b > d)
            {
                // Shrink the existing mapping.
                match = key_type(d + 1, b); // (d, b]
            }

            HPX_ASSERT(!((a < c) && (b > d)));
            HPX_ASSERT(!((a >= c) && (b <= d)));

            ++matches.first;
        }

        value_type* node = new value_type(key, boost::forward<T>(data));
        // TODO: Check for insertion failure?
        return merge(map_.insert(*node).first);
    }

    iterator bind(
        value_type* node
        )
    {
        key_type const& key = node->key_;

        std::pair<iterator, iterator> matches = find(key);

        if (matches.first == end() && matches.second == end())
        {
            std::pair<iterator, bool> r = map_.insert(*node);
            HPX_ASSERT(r.second);
            return merge(r.first);
        }

        for (; matches.first != matches.second;)
        {
            key_type& match = matches.first->key_;

            // Is key a subset of match?
            if (boost::icl::contains(match, key))
                return remap(matches.first, node);

            // Is match a subset of key?
            if (boost::icl::contains(key, match))
            {
                map_.erase_and_dispose(matches.first++, disposer());
                continue;
            }

            // match - [a, b]
            // key   - [c, d]
            Key const a = boost::icl::lower(match);
            Key const b = boost::icl::upper(match);
            Key const c = boost::icl::lower(key);
            Key const d = boost::icl::upper(key);

            // Does match begin before key?
            // |a  b|
            //   |c  d|
            if (a < c)
            {
                // a < c, so c != 0
                HPX_ASSERT(c != 0);

                // Shrink the existing mapping.
                match = key_type(a, c - 1); // [a, c)
            }

            // Does key end before match?
            //   |a  b|
            // |c  d|
            else if (b > d)
            {
                // Shrink the existing mapping.
                match = key_type(d + 1, b); // (d, b]
            }

            HPX_ASSERT(!((a < c) && (b > d)));
            HPX_ASSERT(!((a >= c) && (b <= d)));

            ++matches.first;
        }

        // TODO: Check for insertion failure?
        return merge(map_.insert(*node).first);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    /// Call \p f on the data mapped to [\p lower, \p upper]. For any subsets
    /// of [\p lower, \p upper] that are not mapped, a copy of \p default is
    /// inserted and \p f is applied to it. \p f may be called 0 or more times.
    /// The order in which \p f is called is unspecified, and may not be
    /// sequential. Merges any newly created or updated mappings if possible.
    /// Overwrites or splits other mappings as needed.
    template <
        typename F
    >
    void apply(
        Key const& lower
      , Key const& upper
      , F f
      , data_type const& default_ = data_type()
        )
    {
        key_type const key(lower, upper);
        apply(key, f, default_);
        return;
    }

    /// Call \p f on the data mapped to [\p lower, \p upper]. For any subsets
    /// of \p key that are not mapped, a copy of \p default_ is inserted and
    /// \p f is applied to it. \p f may be called 0 or more times. The order
    /// in which \p f is called is unspecified, and may not be sequential.
    /// Merges any newly created or updated mappings if possible. Overwrites
    /// or splits other mappings as needed.
    template <
        typename F
    >
    void apply(
        key_type const& key
      , F f
      , data_type const& default_ = data_type()
        )
    { // {{{
        std::pair<iterator, iterator> matches = find(key);

        if (matches.first == end() && matches.second == end())
        {
            // Insert a new mapping with default constructed data.
            value_type* node = new value_type(key, default_);
            std::pair<iterator, bool> r = map_.insert(*node);
            HPX_ASSERT(r.second);

            // Call f on the new mapping's data.
            f(node->data_);

            // Try to merge the new mapping.
            merge(r.first);
            return;
        }

        std::list<value_type*> save_list;

        for (; matches.first != matches.second;)
        {
            key_type& match = matches.first->key_;

            // Is key a subset of match?
            if (boost::icl::contains(match, key))
            {
                // Construct an instance of data_type and initialize it with
                // the value of the parent mapping.
                data_type tmp = matches.first->data_;

                // Call f on the temporary.
                f(tmp);

                // Create a new mapping and remap the super object.
                iterator it = remap(matches.first, key, tmp);

                // Try to merge the new mapping.
                merge(it);
                return;
            }

            // Is match a subset of key?
            if (boost::icl::contains(key, match))
            {
                // Save this match, we'll reinsert it later.
                value_type& v = *matches.first;
                save_list.push_back(&v);

                // Call f on the data of the match.
                f(v.data_);

                // Remove the mapping from the list but do not destroy the node.
                map_.erase(matches.first++);
                continue;
            }

            // match - [a, b]
            // key   - [c, d]
            Key const a = boost::icl::lower(match);
            Key const b = boost::icl::upper(match);
            Key const c = boost::icl::lower(key);
            Key const d = boost::icl::upper(key);

            // Does match begin before key?
            // |a  b|
            //   |c  d|
            if (a < c)
            {
                // a < c, so c != 0
                HPX_ASSERT(c != 0);

                // Shrink the existing mapping.
                match = key_type(a, c - 1); // [a, c)

                // Create a new mapping that covers the intersection of the
                // match and the key.
                key_type const intersect(c, b); // [c, b]
                value_type* node
                    = new value_type(intersect, matches.first->data_);

                // Call f on the data of the new mapping.
                f(node->data_);

                // Add a new mapping with the existing mapping's data to the
                // save list.
                save_list.push_back(node);
            }

            // Does key end before match?
            //   |a  b|
            // |c  d|
            else if (b > d)
            {
                // Shrink the existing mapping.
                match = key_type(d + 1, b); // (d, b]

                // Create a new mapping that covers the intersection of the
                // match and the key.
                key_type const intersect(a, d); // [a, d]
                value_type* node
                    = new value_type(intersect, matches.first->data_);

                // Call f on the data of the new mapping.
                f(node->data_);

                // Add a new mapping with the existing mapping's data to the
                // save list.
                save_list.push_back(node);
            }

            HPX_ASSERT(!((a < c) && (b > d)));
            HPX_ASSERT(!((a >= c) && (b <= d)));

            ++matches.first;
        }


        // We've cleared out all the mappings that intersect with key, so now
        // we can insert the "baseline", a mapping that spans the entire key
        // with default constructed data that f has been applied to.
        value_type* node = new value_type(key, default_);

        // Apply f to the baseline mapping.
        f(node->data_);

        // Insert and merge the baseline.
        // TODO: Check for insertion failure?
        merge(map_.insert(*node).first);

        typename std::list<value_type*>::iterator it = save_list.begin()
                                                , e = save_list.end();

        // Now, bind the saved entries, placing them "on top" of the baseline.
        for (; it != e; ++it)
            bind(*it);
    } // }}}

    ///////////////////////////////////////////////////////////////////////////
    std::pair<iterator, iterator> find(
        Key const& lower
      , Key const& upper
        )
    {
        key_type const key(lower, upper);
        return map_.equal_range(key, polymorphic_less());
    }

    std::pair<const_iterator, const_iterator> find(
        Key const& lower
      , Key const& upper
        ) const
    {
        key_type const key(lower, upper);
        return map_.equal_range(key, polymorphic_less());
    }

    template <
        typename T
    >
    std::pair<iterator, iterator> find(
        T const& key
        )
    {
        return map_.equal_range(key, polymorphic_less());
    }

    template <
        typename T
    >
    std::pair<const_iterator, const_iterator> find(
        T const& key
        ) const
    {
        return map_.equal_range(key, polymorphic_less());
    }

    ///////////////////////////////////////////////////////////////////////////
    iterator begin()
    {
        return map_.begin();
    }

    const_iterator begin() const
    {
        return map_.begin();
    }

    iterator end()
    {
        return map_.end();
    }

    const_iterator end() const
    {
        return map_.end();
    }

    ///////////////////////////////////////////////////////////////////////////
    bool empty() const
    {
        return map_.empty();
    }

    size_type size() const
    {
        return map_.size();
    }

    ///////////////////////////////////////////////////////////////////////////
    void clear()
    {
        map_.clear_and_dispose(disposer());
    }

    ///////////////////////////////////////////////////////////////////////////
    size_type erase(
        Key const& lower
      , Key const& upper
        )
    {
        key_type const key(lower, upper);
        return map_.erase_and_dispose(key, polymorphic_less(), disposer());
    }

    template <
        typename T
    >
    size_type erase(
        T const& key
        )
    {
        return map_.erase_and_dispose(key, polymorphic_less(), disposer());
    }

    iterator erase(
        const_iterator pos
        )
    {
        return map_.erase_and_dispose(pos, disposer());
    }

    iterator erase(
        iterator pos
        )
    {
        return map_.erase_and_dispose(const_iterator(pos), disposer());
    }
};

template <
    typename T
>
struct incrementer
{
  private:
    T const amount_;

  public:
    explicit incrementer(
        T const& amount
        )
      : amount_(amount)
    {
        HPX_ASSERT(amount);
    }

    incrementer(
        incrementer const& other
        )
      : amount_(other.amount_)
    {}

    void operator()(
        T& v
        ) const
    {
        v += amount_;
    }
};

template <
    typename T
>
struct decrementer
{
  private:
    T const amount_;

  public:
    decrementer(
        T const& amount
        )
      : amount_(amount)
    {
        HPX_ASSERT(amount);
    }

    decrementer(
        decrementer const& other
        )
      : amount_(other.amount_)
    {}

    void operator()(
        T& v
        ) const
    {
        v -= amount_;
    }
};

}}

#endif // HPX_0097965D_313C_40DB_ABD0_5EA9537DDF8F

