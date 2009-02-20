//  Copyright (C) 2009 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(BOOST_LOCKFREE_DOUBLE_TAGGED_PTR_DCAS_FEB_19_2009_0439PM)
#define BOOST_LOCKFREE_DOUBLE_TAGGED_PTR_DCAS_FEB_19_2009_0439PM

#include <boost/lockfree/cas.hpp>
#include <boost/lockfree/branch_hints.hpp>

#include <cstddef>              /* for std::size_t */

# if defined(BOOST_LOCKFREE_IDENTIFY_CAS_METHOD)
#warning "using tagged_ptr_dcas"
#endif

///////////////////////////////////////////////////////////////////////////////
namespace boost { namespace lockfree
{
    template <class T>
    class BOOST_LOCKFREE_DCAS_ALIGNMENT double_tagged_ptr
    {
    private:
        typedef boost::uint16_t flag_t;

        static const flag_t flag_mask = 1;

        static T* pack_ptr(T* ptr, bool flag = false)
        {
            BOOST_ASSERT(0 == (std::size_t(ptr) & flag_mask));
            return flag ? (T*)(std::size_t(ptr) | flag_mask) : ptr;
        }

        static T* extract_ptr(T* p)
        {
            return (T*)(std::size_t(p) & ~flag_mask);
        }

        static bool extract_flag(T const* i)
        {
            return (std::size_t(i) & flag_mask) ? true : false;
        }

    public:
        typedef std::size_t tag_t;

        struct ptr_tag
        {
            ptr_tag() : ptr_(0), tag_(0) {}
            ptr_tag(T* p, tag_t t = false) : ptr_(p), tag_(t) {}

            void set(T* p, tag_t t) 
            {
                ptr_ = p;
                tag_ = t;
            }

            boost::uint64_t const& as_uint64() const
            {
                BOOST_ASSERT(sizeof(boost::uint64_t) == sizeof(ptr_tag));
                return *reinterpret_cast<boost::uint64_t const*>(this);
            }

            T* get_ptr() { return extract_ptr(ptr_); }

            bool CAS(ptr_tag const& oldval, T* newptr)
            {
                return boost::lockfree::CAS2(this, oldval.ptr_, oldval.tag_, 
                    newptr, oldval.tag_ + 1);
            }

            friend bool operator== (ptr_tag const& lhs, ptr_tag const& rhs)
            {
                return lhs.ptr_ == rhs.ptr_ && lhs.tag_ == rhs.tag_;
            }
            friend bool operator!= (ptr_tag const& lhs, ptr_tag const& rhs)
            {
                return !(lhs == rhs);
            }
            friend bool operator== (ptr_tag volatile const& lhs, ptr_tag const& rhs)
            {
                return lhs.ptr_ == rhs.ptr_ && lhs.tag_ == rhs.tag_;
            }
            friend bool operator!= (ptr_tag volatile const& lhs, ptr_tag const& rhs)
            {
                return !(lhs == rhs);
            }

            T* ptr_;
            tag_t tag_;
        };

    public:
        /** uninitialized constructor */
        double_tagged_ptr() 
        {}

        /** copy constructor */
        double_tagged_ptr(double_tagged_ptr const & p)
        {
            set(p);
        }

        explicit double_tagged_ptr(T* pl, T* pr, tag_t tl = 0, tag_t tr = 0)
          : left_(pl, tl), right_(pr, tr)
        {}

        double_tagged_ptr(ptr_tag const& l, ptr_tag const& r)
          : left_(l), right_(r)
        {}

        /** atomic set operations */
        /* @{ */
        void operator= (double_tagged_ptr const& p)
        {
            atomic_set(p);
        }

        void atomic_set(double_tagged_ptr const& p)
        {
            for (;;)
            {
                double_tagged_ptr old;
                old.set(*this);

                if(likely(CAS(old, p.left_, p.right_)))
                    return;
            }
        }
        /* @} */

        friend ptr_tag make_unique(ptr_tag const& p)
        {
            if (extract_ptr(p.ptr_) != 0)
                return ptr_tag(p.ptr_, p.tag_ + 1);
            return ptr_tag();
        }

        /** unsafe set operation */
        /* @{ */
        void set(double_tagged_ptr const & p)
        {
            left_ = p.left_;
            right_ = p.right_;
        }

        void set(T* pl, T* pr, tag_t tl, tag_t tr)
        {
            left_.set(pl, tl);
            right_.set(pr, tr);
        }
        /* @} */

        /** comparing semantics */
        /* @{ */
        friend bool operator== (double_tagged_ptr const& lhs, double_tagged_ptr const& rhs)
        {
            return lhs.left_ == rhs.left_ && lhs.right_ == rhs.right_;
        }
        friend bool operator!= (double_tagged_ptr const& lhs, double_tagged_ptr const& rhs)
        {
            return !(lhs == rhs);
        }
        friend bool operator== (double_tagged_ptr volatile const& lhs, double_tagged_ptr const& rhs)
        {
            return lhs.left_ == rhs.left_ && lhs.right_ == rhs.right_;
        }
        friend bool operator!= (double_tagged_ptr volatile const& lhs, double_tagged_ptr const& rhs)
        {
            return !(lhs == rhs);
        }
        /* @} */

        /** pointer access */
        /* @{ */
        T* get_left_ptr() const
        {
            return extract_ptr(left_.ptr_);
        }
        T* get_right_ptr() const
        {
            return extract_ptr(right_.ptr_);
        }

        void set_left_ptr(T* p)
        {
            bool f = extract_flag(left_.ptr_);
            left_.ptr_ = pack_ptr(p, f);
        }
        void set_right_ptr(T* p)
        {
            bool f = extract_flag(right_.ptr_);
            right_.ptr_ = pack_ptr(p, f);
        }
        /* @} */

        /** tag access */
        /* @{ */
        tag_t get_left_tag() const
        {
            return left_.tag_;
        }
        tag_t get_right_tag() const
        {
            return right_.tag_;
        }

        void set_left_tag(tag_t t)
        {
            left_.tag_ = t;
        }
        void set_right_tag(tag_t t)
        {
            right_.tag_ = t;
        }
        /* @} */

        /** flag access */
        /* @{ */
        bool get_left_flag() const
        {
            return extract_flag(left_.ptr_);
        }
        bool get_right_flag() const
        {
            return extract_flag(right_.ptr_);
        }

        void set_left_flag(bool flag = true)
        {
            T* p = get_left_ptr();
            left_.ptr_ = pack_ptr(p, flag);
        }
        void set_right_flag(bool flag = true)
        {
            T* p = get_right_ptr();
            right_.ptr_ = pack_ptr(p, flag);
        }
        /* @} */

        /** compare and swap  */
        /* @{ */
        bool CAS(double_tagged_ptr const & oldval, ptr_tag const& newleft, 
            ptr_tag const& newright)
        {
            return boost::lockfree::CAS2(this, oldval.left_.as_uint64(), 
                oldval.right_.as_uint64(), newleft.as_uint64(), newright.as_uint64());
        }
        bool CAS(double_tagged_ptr const& oldval, double_tagged_ptr& newval)
        {
            newval.left_.tag_ = oldval.left_.tag_ + 1;
            newval.right_.tag_ = oldval.right_.tag_ + 1;
            return boost::lockfree::CAS2(this, 
                oldval.left_.as_uint64(), oldval.right_.as_uint64(), 
                newval.left_.as_uint64(), newval.right_.as_uint64());
        }
        /* @} */

        /** smart pointer support  */
        /* @{ */
    private:
        void bool_stub(T*) {}
        typedef void (double_tagged_ptr::*bool_type)(T*);

    public:
        operator bool_type() const
        {
            return (0 != get_left_ptr() && 0 != get_right_ptr()) ? 
                &double_tagged_ptr::bool_stub : NULL;
        }
        /* @} */

        ptr_tag left_;
        ptr_tag right_;
    };

}} // boost::lockfree

#endif 
