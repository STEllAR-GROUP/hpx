////////////////////////////////////////////////////////////////////////////////
//  Algorithms from "CAS-Based Lock-Free Algorithm for Shared Deques"
//  by M. M. Michael
//  Link: http://www.research.ibm.com/people/m/michael/europar-2003.pdf
//
//  C++ implementation - Copyright (C) 2011 Bryce Lelbach
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
//  Disclaimer: Not a Boost library.
//
//  The complexity of each operation is defined in terms of total work (see
//  section 2.5 of the paper) under contention from P processes. Without
//  contention, all operations have a constant complexity, except for the dtor.
////////////////////////////////////////////////////////////////////////////////

#if !defined(HPX_F985C12D_03E7_4E25_8CB1_018A56A265E0)
#define HPX_F985C12D_03E7_4E25_8CB1_018A56A265E0

#include <hpx/config.hpp>

#include <boost/thread/thread.hpp>
#include <iostream>

#include <boost/atomic.hpp>

#include <boost/lockfree/detail/tagged_ptr.hpp>

#include <hpx/util/lockfree/detail/tagged_ptr_pair.hpp>
#include <hpx/util/lockfree/freelist.hpp>

#include <type_traits>

namespace boost { namespace lockfree
{

// The "left" and "right" terminology is used instead of top and bottom to stay
// consistent with the paper that this code is based on..
enum deque_status_type
{
    stable,
    rpush,
    lpush
};

template <typename T>
struct deque_node //-V690
{
    typedef detail::tagged_ptr<deque_node> pointer;
    typedef atomic<pointer> atomic_pointer;

    typedef typename pointer::tag_t tag_t;

    atomic_pointer left;
    atomic_pointer right;
    T data;

    deque_node(): left(0), right(0), data() {}

    deque_node(deque_node const& p):
        left(p.left.load()), right(p.right.load()) {}

    deque_node(deque_node* lptr, deque_node* rptr, T const& v,
               tag_t ltag = 0, tag_t rtag = 0):
        left(pointer(lptr, ltag)), right(pointer(rptr, rtag)), data(v) {}
};

// FIXME: A lot of these methods can be dropped; in fact, it may make sense to
// re-structure this class like deque_node.
template <typename T>
struct deque_anchor //-V690
{
    typedef deque_node<T> node;
    typedef typename node::pointer node_pointer;
    typedef typename node::atomic_pointer atomic_node_pointer;

    typedef typename node::tag_t tag_t;

    typedef tagged_ptr_pair<node, node> pair;
    typedef boost::atomic<pair> atomic_pair;

  private:
    atomic_pair pair_;

  public:
    deque_anchor(): pair_(pair(nullptr, nullptr, stable, 0)) {}

    deque_anchor(deque_anchor const& p): pair_(p.pair_.load()) {}

    deque_anchor(pair const& p): pair_(p) {}

    deque_anchor(node* lptr, node* rptr,
                 tag_t status = stable, tag_t tag = 0):
        pair_(pair(lptr, rptr, status, tag)) {}

    pair lrs() const volatile
    { return pair_.load(); }

    node* left() const volatile
    { return pair_.load().get_left_ptr(); }

    node* right() const volatile
    { return pair_.load().get_right_ptr(); }

    tag_t status() const volatile
    { return pair_.load().get_left_tag(); }

    tag_t tag() const volatile
    { return pair_.load().get_right_tag(); }

    bool cas(deque_anchor& expected, deque_anchor const& desired) volatile
    { return pair_.compare_exchange_strong(expected.load(), desired.load()); }

    bool cas(pair& expected, deque_anchor const& desired) volatile
    { return pair_.compare_exchange_strong(expected, desired.load()); }

    bool cas(deque_anchor& expected, pair const& desired) volatile
    { return pair_.compare_exchange_strong(expected.load(), desired); }

    bool cas(pair& expected, pair const& desired) volatile
    { return pair_.compare_exchange_strong(expected, desired); }

    bool operator==(volatile deque_anchor const& rhs) const
    { return pair_.load() == rhs.pair_.load(); }

    bool operator!=(volatile deque_anchor const& rhs) const
    { return !(*this == rhs); }

    bool operator==(volatile pair const& rhs) const
    { return pair_.load() == rhs; }

    bool operator!=(volatile pair const& rhs) const
    { return !(*this == rhs); }

    bool is_lock_free() const
    { return pair_.is_lock_free(); }
};

// TODO: Experiment with memory ordering to see where we can optimize without
// breaking things.
template <typename T,
          typename freelist_t = caching_freelist_t,
          typename Alloc = std::allocator<T>
          >
struct deque
{
  private:
    HPX_NON_COPYABLE(deque);

  public:
    typedef deque_node<T> node;
    typedef typename node::pointer node_pointer;
    typedef typename node::atomic_pointer atomic_node_pointer;

    typedef typename node::tag_t tag_t;

    typedef deque_anchor<T> anchor;
    typedef typename anchor::pair anchor_pair;
    typedef typename anchor::atomic_pair atomic_anchor_pair;

    typedef typename Alloc::template rebind<node>::other node_allocator;

    typedef typename std::conditional<
        std::is_same<freelist_t, caching_freelist_t>::value,
        caching_freelist<node, node_allocator>,
        static_freelist<node, node_allocator>
    >::type pool;

  private:
    anchor anchor_;
    pool pool_;

    HPX_STATIC_CONSTEXPR int
        padding_size = BOOST_LOCKFREE_CACHELINE_BYTES - sizeof(anchor); //-V103
    char padding[padding_size];

    node* alloc_node(node* lptr, node* rptr, T const& v,
                     tag_t ltag = 0, tag_t rtag = 0)
    {
        node* chunk = pool_.allocate();
        new (chunk) node(lptr, rptr, v, ltag, rtag);
        return chunk;
    }

    void dealloc_node(node* n)
    {
        n->~node();
        pool_.deallocate(n);
    }

    void stabilize_left(anchor_pair& lrs)
    {
        // Get the right node of the leftmost pointer held by lrs and its ABA
        // tag (tagged_ptr).
        node_pointer prev = lrs.get_left_ptr()->right.load();

        if (anchor_ != lrs)
            return;

        // Get the left node of prev and its tag (again, a tuple represented by
        // a tagged_ptr).
        node_pointer prevnext = prev.get_ptr()->left.load();

        // Check if prevnext is equal to r.
        if (prevnext.get_ptr() != lrs.get_left_ptr())
        {
            if (anchor_ != lrs)
                return;

            // Attempt the CAS, incrementing the tag to protect from the ABA
            // problem.
            if (!prev.get_ptr()->left.compare_exchange_strong(prevnext,
                     node_pointer(lrs.get_left_ptr(), prevnext.get_tag() + 1)))
                return;
        }
        // Try to update the anchor, modifying the status and ABA tag.
        anchor_.cas(lrs, anchor_pair(lrs.get_left_ptr(), lrs.get_right_ptr(),
             stable, lrs.get_right_tag() + 1));
    }

    void stabilize_right(anchor_pair& lrs)
    {
        // Get the left node of the rightmost pointer held by lrs and its ABA
        // tag (tagged_ptr).
        node_pointer prev = lrs.get_right_ptr()->left.load();

        if (anchor_ != lrs)
            return;

        // Get the right node of prev and its tag (again, a tuple represented
        // by a tagged_ptr).
        node_pointer prevnext = prev.get_ptr()->right.load();

        // Check if prevnext is equal to r.
        if (prevnext.get_ptr() != lrs.get_right_ptr())
        {
            if (anchor_ != lrs)
                return;

            // Attempt the CAS, incrementing the tag to protect from the ABA
            // problem.
            if (!prev.get_ptr()->right.compare_exchange_strong(prevnext,
                     node_pointer(lrs.get_right_ptr(), prevnext.get_tag() + 1)))
                return;
        }
        // Try to update the anchor, modifying the status and ABA tag.
        anchor_.cas(lrs, anchor_pair(lrs.get_left_ptr(), lrs.get_right_ptr(),
             stable, lrs.get_right_tag() + 1));
    }

    void stabilize(anchor_pair& lrs)
    {
        // The left tag stores the status.
        if (lrs.get_left_tag() == rpush)
            stabilize_right(lrs);
        else // lrs.s() == lpush
            stabilize_left(lrs);
    }

  public:
    deque(std::size_t initial_nodes = 128): anchor_(), pool_(initial_nodes) {}

    // Not thread-safe.
    // Complexity: O(N*Processes)
    ~deque()
    {
        if (!empty())
        {
            T dummy = T();
            while (true)
            {
                if (!pop_left(dummy))
                    break;
            }
        }
    }

    // Not thread-safe.
    // Complexity: O(Processes)
    // FIXME: Should we check both pointers here?
    bool empty() const
    { return anchor_.lrs().get_left_ptr() == nullptr; }

    // Thread-safe and non-blocking.
    // Complexity: O(1)
    bool is_lock_free() const
    { return anchor_.is_lock_free(); }

    // Thread-safe and non-blocking (may block if node needs to be allocated
    // from the operating system). Returns false if the freelist is not able to
    // allocate a new deque node.
    // Complexity: O(Processes)
    bool push_left(T const& data)
    {
        // Allocate the new node which we will be inserting.
        node* n = alloc_node(nullptr, nullptr, data);

        if (n == nullptr)
            return false;

        // Loop until we insert successfully.
        while (true)
        {
            // Load the anchor.
            anchor_pair lrs = anchor_.lrs();

            // Check if the deque is empty.
            // FIXME: Should we check both pointers here?
            if (lrs.get_left_ptr() == nullptr)
            {
                // If the deque is empty, we simply install a new anchor which
                // points to the new node as both its leftmost and rightmost
                // element.
                if (anchor_.cas(lrs, anchor_pair(n, n,
                        lrs.get_left_tag(), lrs.get_right_tag() + 1)))
                    return true;
            }

            // Check if the deque is stable.
            else if (lrs.get_left_tag() == stable)
            {
                // Make the right pointer on our new node refer to the current
                // leftmost node.
                n->right.store(node_pointer(lrs.get_left_ptr()));

                // Now we want to make the anchor point to our new node as the
                // leftmost node. We change the state to lpush as the deque
                // will become unstable if this operation succeeds.
                anchor_pair new_anchor(n, lrs.get_right_ptr(),
                    lpush, lrs.get_right_tag() + 1);

                if (anchor_.cas(lrs, new_anchor))
                {
                    stabilize_left(new_anchor);
                    return true;
                }
            }

            // The deque must be unstable, so we have to stabilize it before
            // we can continue.
            else // lrs.s() != stable
                stabilize(lrs);
        }
    }

    // Thread-safe and non-blocking (may block if node needs to be allocated
    // from the operating system). Returns false if the freelist is not able to
    // allocate a new deque node.
    // Complexity: O(Processes)
    bool push_right(T const& data)
    {
        // Allocate the new node which we will be inserting.
        node* n = alloc_node(0, 0, data);

        if (n == 0)
            return false;

        // Loop until we insert successfully.
        while (true)
        {
            // Load the anchor.
            anchor_pair lrs = anchor_.lrs();

            // Check if the deque is empty.
            // FIXME: Should we check both pointers here?
            if (lrs.get_right_ptr() == 0)
            {
                // If the deque is empty, we simply install a new anchor which
                // points to the new node as both its leftmost and rightmost
                // element.
                if (anchor_.cas(lrs, anchor_pair(n, n,
                        lrs.get_left_tag(), lrs.get_right_tag() + 1)))
                    return true;
            }

            // Check if the deque is stable.
            else if (lrs.get_left_tag() == stable)
            {
                // Make the left pointer on our new node refer to the current
                // rightmost node.
                n->left.store(node_pointer(lrs.get_right_ptr()));

                // Now we want to make the anchor point to our new node as the
                // leftmost node. We change the state to lpush as the deque
                // will become unstable if this operation succeeds.
                anchor_pair new_anchor(lrs.get_left_ptr(), n,
                    rpush, lrs.get_right_tag() + 1);

                if (anchor_.cas(lrs, new_anchor))
                {
                    stabilize_right(new_anchor);
                    return true;
                }
            }

            // The deque must be unstable, so we have to stabilize it before
            // we can continue.
            else // lrs.s() != stable
                stabilize(lrs);
        }
    }

    // Thread-safe and non-blocking. Returns false if the deque is empty.
    // Complexity: O(Processes)
    bool pop_left(T& r)
    {
        // Loop until we either pop an element or learn that the deque is empty.
        while (true)
        {
            // Load the anchor.
            anchor_pair lrs = anchor_.lrs();

            // Check if the deque is empty.
            // FIXME: Should we check both pointers here?
            if (lrs.get_left_ptr() == nullptr)
                return false;

            // Check if the deque has 1 element.
            if (lrs.get_left_ptr() == lrs.get_right_ptr())
            {
                // Try to set both anchor pointer
                if (anchor_.cas(lrs, anchor_pair(nullptr, nullptr,
                        lrs.get_left_tag(), lrs.get_right_tag() + 1)))
                {
                    // Set the result, deallocate the popped node, and return.
                    r = lrs.get_left_ptr()->data;
                    dealloc_node(lrs.get_left_ptr());
                    return true;
                }
            }

            // Check if the deque is stable.
            else if (lrs.get_left_tag() == stable)
            {
                // Make sure the anchor hasn't changed since we loaded it.
                if (anchor_ != lrs)
                    continue;

                // Get the leftmost nodes' right node.
                node_pointer prev = lrs.get_left_ptr()->right.load();

                // Try to update the anchor to point to prev as the leftmost
                // node.
                if (anchor_.cas(lrs, anchor_pair(prev.get_ptr(),
                        lrs.get_right_ptr(), lrs.get_left_tag(),
                        lrs.get_right_tag() + 1)))
                {
                    // Set the result, deallocate the popped node, and return.
                    r = lrs.get_left_ptr()->data;
                    dealloc_node(lrs.get_left_ptr());
                    return true;
                }
            }

            // The deque must be unstable, so we have to stabilize it before
            // we can continue.
            else // lrs.s() != stable
                stabilize(lrs);
        }
    }

    bool pop_left(T* r) { return pop_left(*r); }

    // Thread-safe and non-blocking. Returns false if the deque is empty.
    // Complexity: O(Processes)
    bool pop_right(T& r)
    {
        // Loop until we either pop an element or learn that the deque is empty.
        while (true)
        {
            // Load the anchor.
            anchor_pair lrs = anchor_.lrs();

            // Check if the deque is empty.
            // FIXME: Should we check both pointers here?
            if (lrs.get_right_ptr() == nullptr)
                return false;

            // Check if the deque has 1 element.
            if (lrs.get_right_ptr() == lrs.get_left_ptr())
            {
                // Try to set both anchor pointer
                if (anchor_.cas(lrs, anchor_pair(nullptr, nullptr,
                        lrs.get_left_tag(), lrs.get_right_tag() + 1)))
                {
                    // Set the result, deallocate the popped node, and return.
                    r = lrs.get_right_ptr()->data;
                    dealloc_node(lrs.get_right_ptr());
                    return true;
                }
            }

            // Check if the deque is stable.
            else if (lrs.get_left_tag() == stable)
            {
                // Make sure the anchor hasn't changed since we loaded it.
                if (anchor_ != lrs)
                    continue;

                // Get the rightmost nodes' left node.
                node_pointer prev = lrs.get_right_ptr()->left.load();

                // Try to update the anchor to point to prev as the rightmost
                // node.
                if (anchor_.cas(lrs, anchor_pair(lrs.get_left_ptr(),
                        prev.get_ptr(), lrs.get_left_tag(),
                        lrs.get_right_tag() + 1)))
                {
                    // Set the result, deallocate the popped node, and return.
                    r = lrs.get_right_ptr()->data;
                    dealloc_node(lrs.get_right_ptr());
                    return true;
                }
            }

            // The deque must be unstable, so we have to stabilize it before
            // we can continue.
            else // lrs.s() != stable
                stabilize(lrs);
        }
    }

    bool pop_right(T* r) { return pop_right(*r); }
};

}}

#endif // HPX_F985C12D_03E7_4E25_8CB1_018A56A265E0

