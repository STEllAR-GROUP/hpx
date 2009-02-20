//  lock-free dequeue from
//  Michael, M. M., "CAS-Based Lock-Free Algorithm for Shared Deques"
//
//  Copyright (c) 2009 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_LOCKFREE_DEQUE_HPP_INCLUDED
#define BOOST_LOCKFREE_DEQUE_HPP_INCLUDED

#include <boost/lockfree/prefix.hpp>
#include <boost/lockfree/double_tagged_ptr.hpp>
#include <boost/lockfree/atomic_int.hpp>
#include <boost/lockfree/freelist.hpp>

#include <boost/concept_check.hpp>
#include <boost/static_assert.hpp>
#include <boost/noncopyable.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace lockfree
{
    template <typename T, typename Alloc = std::allocator<T> >
    class deque : boost::noncopyable
    {
        BOOST_CLASS_REQUIRE(T, boost, CopyConstructibleConcept);
        BOOST_CLASS_REQUIRE(T, boost, DefaultConstructibleConcept);

        struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT node
        {
            node(T const & v)
              : data_(v)
            {}

            double_tagged_ptr<node> node_;
            T data_;
        };
        typedef typename double_tagged_ptr<node>::ptr_tag atomic_node_ptr;

        struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT anchor
        {
            enum state 
            { 
                stable = 0,
                left_push = 1,
                right_push = 2
            };

            anchor()
            {}

            anchor(node* left, node* right, state s)
              : node_(left, right)
            {
                BOOST_ASSERT(NULL != node_.get_left_ptr() || NULL == node_.get_right_ptr());
                BOOST_ASSERT(NULL != node_.get_right_ptr() || NULL == node_.get_left_ptr());
                set_state(s);
            }

            anchor(atomic_node_ptr const& left, node* right, state s)
              : node_(left, right)
            {
                BOOST_ASSERT(NULL != node_.get_left_ptr() || NULL == node_.get_right_ptr());
                BOOST_ASSERT(NULL != node_.get_right_ptr() || NULL == node_.get_left_ptr());
                set_state(s);
            }

            anchor(node* left, atomic_node_ptr const& right, state s)
              : node_(left, right)
            {
                BOOST_ASSERT(NULL != node_.get_left_ptr() || NULL == node_.get_right_ptr());
                BOOST_ASSERT(NULL != node_.get_right_ptr() || NULL == node_.get_left_ptr());
                set_state(s);
            }

            anchor(atomic_node_ptr const& left, atomic_node_ptr const& right, state s)
              : node_(left, right)
            {
                BOOST_ASSERT(NULL != node_.get_left_ptr() || NULL == node_.get_right_ptr());
                BOOST_ASSERT(NULL != node_.get_right_ptr() || NULL == node_.get_left_ptr());
                set_state(s);
            }

            anchor(anchor const& rhs)
            {
                node_.atomic_set(rhs.node_);
            }
            anchor& operator=(anchor const& rhs)
            {
                if (this != &rhs)
                    node_.atomic_set(rhs.node_);
                return *this;
            }

            state get_state() const
            {
                return node_.get_left_flag() ? left_push : 
                          (node_.get_right_flag() ? right_push : stable);
            }

            void set_state(state s)
            {
                node_.set_left_flag(false);
                node_.set_right_flag(false);
                if (left_push == s)
                    node_.set_left_flag();
                else if (right_push == s)
                    node_.set_right_flag();
            }

            bool CAS (anchor const& oldval, anchor& newval)
            {
                return node_.CAS(oldval.node_, newval.node_);
            }

            friend bool operator== (anchor const& lhs, anchor const& rhs)
            {
                return lhs.node_ == rhs.node_;
            }
            friend bool operator!= (anchor const& lhs, anchor const& rhs)
            {
                return lhs.node_ != rhs.node_;
            }

            double_tagged_ptr<node> node_;
        };

    public:
        deque()
        {}

        explicit deque(std::size_t initial_nodes)
          : pool(initial_nodes)
        {}

        ~deque()
        {
            BOOST_ASSERT(empty());
        }

        bool empty(void) const
        {
            // either root pointers have to be either NULL or not NULL
            BOOST_ASSERT(NULL != root_.node_.get_right_ptr() || 
                         NULL == root_.node_.get_left_ptr());
            return NULL == root_.node_.get_right_ptr();
        }

        void push_right(T const& t)
        {
            node* n = alloc_node(t);
            for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
            {
                anchor root(root_);
                if (NULL == root.node_.get_right_ptr())
                {
                    anchor newroot(n, n, anchor::stable);
                    if (root_.CAS(root, newroot)) 
                        return;
                }
                else if (anchor::stable == root.get_state())
                {
                    n->node_.left_ = root.node_.right_;
                    anchor newroot(root.node_.left_, n, anchor::right_push);
                    if (root_.CAS(root, newroot))
                    {
                        stabilize_right(newroot);
                        return;
                    }
                }
                else
                {
                    stabilize(root);
                }
            }
        }

        void push_left(T const& t)
        {
            node* n = alloc_node(t);
            for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
            {
                anchor root(root_);
                if (NULL == root.node_.get_left_ptr())
                {
                    anchor newroot(n, n, anchor::stable);
                    if (root_.CAS(root, newroot)) 
                        return;
                }
                else if (anchor::stable == root.get_state())
                {
                    n->node_.right_ = root.node_.left_;
                    anchor newroot(n, root.node_.right_, anchor::left_push);
                    if (root_.CAS(root, newroot))
                    {
                        stabilize_left(newroot);
                        return;
                    }
                }
                else
                {
                    stabilize(root);
                }
            }
        }

        bool pop_right(T* ret)
        {
            for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
            {
                anchor root(root_);
                node* right = root.node_.get_right_ptr();
                if (NULL == right)
                    return false;

                if (root.node_.get_left_ptr() == right)
                {
                    anchor newroot(NULL, NULL, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = right->data_;
                        dealloc_node(right);
                        return true;
                    }
                }
                else if (anchor::stable == root.get_state())
                {
                    atomic_node_ptr nextright(right->node_.left_);
                    memory_barrier();
                    if (root_ != root) continue;

                    anchor newroot(root.node_.left_, nextright, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = right->data_;
                        dealloc_node(right);
                        return true;
                    }
                }
                else
                {
                    stabilize(root);
                }
            }
        }

        bool pop_left(T* ret)
        {
            for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
            {
                anchor root(root_);
                node* left = root.node_.get_left_ptr();
                if (NULL == left)
                    return false;

                if (left == root.node_.get_right_ptr())
                {
                    anchor newroot(NULL, NULL, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = left->data_;
                        dealloc_node(left);
                        return true;
                    }
                }
                else if (anchor::stable == root.get_state())
                {
                    atomic_node_ptr nextleft(left->node_.right_);
                    memory_barrier();
                    if (root_ != root) continue;

                    anchor newroot(nextleft, root.node_.right_, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = left->data_;
                        dealloc_node(left);
                        return true;
                    }
                }
                else
                {
                    stabilize(root);
                }
            }
        }

    protected:
        void stabilize(anchor const& root)
        {
            BOOST_ASSERT(anchor::stable != root.get_state());
            if (anchor::left_push == root.get_state())
                stabilize_left(root);
            else
                stabilize_right(root);
        }

        void stabilize_right(anchor const& root)
        {
            atomic_node_ptr prev(root.node_.get_right_ptr()->node_.left_);
            memory_barrier();
            if (root_ != root) return;

            atomic_node_ptr prevright(prev.get_ptr()->node_.right_);
            memory_barrier();
            if (prevright.get_ptr() != root.node_.get_right_ptr())
            {
                if (root_ != root) 
                    return;
                if (!prev.get_ptr()->node_.right_.CAS(prevright, root.node_.get_right_ptr()))
                    return;
            }

            // make the root stable
            anchor newroot(root.node_.left_, root.node_.right_, anchor::stable);
            root_.CAS(root, newroot);
        }

        void stabilize_left(anchor const& root)
        {
            atomic_node_ptr prev(root.node_.get_left_ptr()->node_.right_);
            memory_barrier();
            if (root_ != root) return;

            atomic_node_ptr prevleft(prev.get_ptr()->node_.left_);
            memory_barrier();
            if (prevleft.get_ptr() != root.node_.get_left_ptr())
            {
                if (root_ != root) 
                    return;
                if (!prev.get_ptr()->node_.left_.CAS(prevleft, root.node_.get_left_ptr()))
                    return;
            }

            // make the root stable
            anchor newroot(root.node_.left_, root.node_.right_, anchor::stable);
            root_.CAS(root, newroot);
        }

    private:
        node* alloc_node(T const& t)
        {
            node* chunk = pool.allocate();
            new(chunk) node(t);
            return chunk;
        }

        void dealloc_node(node* n)
        {
            n->~node();
            pool.deallocate(n);
        }

        typedef typename Alloc::template rebind<node>::other node_allocator;
        boost::lockfree::caching_freelist<node, node_allocator> pool;

        anchor root_;
    };

}}

#endif

