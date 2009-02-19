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
#include <boost/lockfree/tagged_ptr.hpp>
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
              : data(v), left_(NULL), right_(NULL)
            {}

            node()
              : left_(NULL), right_(NULL)
            {}

            tagged_ptr<node> left_;
            tagged_ptr<node> right_;
            T data;
        };
        typedef tagged_ptr<node> atomic_node_ptr;

        struct BOOST_LOCKFREE_CACHELINE_ALIGNMENT anchor
        {
            enum state { 
                stable = 0,
                left_push = 1,
                right_push = 2
            };

            anchor()
              : left_(NULL), right_(NULL)
            {}

            anchor(node* left, node* right, state s)
              : left_(left), right_(right)
            {
                set_state(s);
            }

            anchor(atomic_node_ptr left, node* right, state s)
              : left_(make_unique(left)), right_(right)
            {
                set_state(s);
            }

            anchor(node* left, atomic_node_ptr right, state s)
              : left_(left), right_(make_unique(right))
            {
                set_state(s);
            }

            anchor(atomic_node_ptr left, atomic_node_ptr right, state s)
              : left_(make_unique(left)), right_(make_unique(right))
            {
                set_state(s);
            }

            state get_state() const
            {
                return left_.get_flag() ? left_push : 
                          (right_.get_flag() ? right_push : stable);
            }

            void set_state(state s)
            {
                left_.set_flag(false);
                right_.set_flag(false);
                if (left_push == s)
                    left_.set_flag();
                else if (right_push == s)
                    right_.set_flag();
            }

            bool CAS (anchor const& oldval, anchor const& newval)
            {
                return boost::lockfree::CAS2(
                    this, oldval.left_, oldval.right_, newval.left_, newval.right_);
            }

            friend bool operator== (anchor const& lhs, anchor const& rhs)
            {
                return lhs.left_ == rhs.left_ && lhs.right_ == rhs.right_;
            }
            friend bool operator!= (anchor const& lhs, anchor const& rhs)
            {
                return !(lhs == rhs);
            }

            tagged_ptr<node> left_;
            tagged_ptr<node> right_;
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
            BOOST_ASSERT(NULL != root_.right_.get_ptr() || 
                         NULL == root_.left_.get_ptr());
            return NULL == root_.right_.get_ptr();
        }

        void push_right(T const& t)
        {
            node* n = alloc_node(t);
            for (unsigned int cnt = 0; /**/; spin((unsigned char)++cnt))
            {
                anchor root(root_);
                memory_barrier();

                if (NULL == root.right_.get_ptr())
                {
                    anchor newroot(n, n, anchor::stable);
                    if (root_.CAS(root, newroot)) 
                        return;
                }
                else if (anchor::stable == root.get_state())
                {
                    n->left_ = make_unique(root.right_);
                    anchor newroot(root.left_, n, anchor::right_push);
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
                memory_barrier();

                if (NULL == root.left_.get_ptr())
                {
                    anchor newroot(n, n, anchor::stable);
                    if (root_.CAS(root, newroot)) 
                        return;
                }
                else if (anchor::stable == root.get_state())
                {
                    n->right_ = make_unique(root.left_);
                    anchor newroot(n, root.right_, anchor::left_push);
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
                memory_barrier();

                node* right = root.right_.get_ptr();
                if (NULL == right)
                    return false;

                if (right == root.left_.get_ptr())
                {
                    anchor newroot(NULL, NULL, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = right->data;
                        dealloc_node(right);
                        return true;
                    }
                }
                else if (anchor::stable == root.get_state())
                {
                    anchor newroot(root.left_, right->left_, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = right->data;
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
                memory_barrier();

                node* left = root.left_.get_ptr();
                if (NULL == left)
                    return false;

                if (root.right_.get_ptr() == left)
                {
                    anchor newroot(NULL, NULL, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = left->data;
                        dealloc_node(left);
                        return true;
                    }
                }
                else if (anchor::stable == root.get_state())
                {
                    anchor newroot(left->right_, root.right_, anchor::stable);
                    if (root_.CAS(root, newroot))
                    {
                        *ret = left->data;
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
            atomic_node_ptr prev(root.right_.get_ptr()->left_);
            memory_barrier();
            if (root_ != root) return;

            atomic_node_ptr prevright(prev.get_ptr()->right_);
            memory_barrier();
            if (prevright.get_ptr() != root.right_.get_ptr())
            {
                if (root_ != root) 
                    return;
                if (!prev.get_ptr()->right_.CAS(prevright, root.right_.get_ptr()))
                    return;
            }

            // make the root stable
            anchor newroot(root.left_, root.right_, anchor::stable);
            root_.CAS(root, newroot);
        }

        void stabilize_left(anchor const& root)
        {
            atomic_node_ptr prev(root.left_.get_ptr()->right_);
            memory_barrier();
            if (root_ != root) return;

            atomic_node_ptr prevleft(prev.get_ptr()->left_);
            memory_barrier();
            if (prevleft.get_ptr() != root.left_.get_ptr())
            {
                if (root_ != root) 
                    return;
                if (!prev.get_ptr()->left_.CAS(prevleft, root.left_.get_ptr()))
                    return;
            }

            // make the root stable
            anchor newroot(root.left_, root.right_, anchor::stable);
            root_.CAS(root, newroot);
        }

    private:
        node * alloc_node(void)
        {
            node* chunk = pool.allocate();
            new(chunk) node();
            return chunk;
        }

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

