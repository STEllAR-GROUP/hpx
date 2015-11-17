//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TAGGED_THREAD_STATE_MAR_12_2010_0125PM)
#define HPX_TAGGED_THREAD_STATE_MAR_12_2010_0125PM

#include <boost/cstdint.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    template <typename T>
    class tagged_thread_state
    {
        typedef boost::int32_t tagged_thread_state_type;
        typedef boost::int8_t thread_state_type;
        typedef boost::int32_t tag_type;

    private:
        union cast_type
        {
            tagged_thread_state_type value;
            thread_state_type tag[sizeof(boost::int32_t)];
        };

        static const int state_index = sizeof(boost::int32_t)-1;
        static const tagged_thread_state_type tag_mask = 0xffffff;  // (1L<<24L)-1;

        static tag_type
        extract_tag(volatile tagged_thread_state_type const& i)
        {
            return i & tag_mask;    // blend out state
        }

        static thread_state_type
        extract_state(volatile tagged_thread_state_type const& i)
        {
            cast_type cu;
            cu.value = i;
            return cu.tag[state_index];
        }

        static tagged_thread_state_type
        pack_state(tagged_thread_state_type state, tag_type tag)
        {
            cast_type ret;
            ret.value = tagged_thread_state_type(tag);
            ret.tag[state_index] = thread_state_type(state);
            return ret.value;
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        tagged_thread_state() : state_(0) {}

//         tagged_thread_state(tagged_thread_state const& p)
//           : state_(p.state_)
//         {}

        explicit tagged_thread_state(T state, tag_type t = 0)
          : state_(pack_state(state, t))
        {}

        ///////////////////////////////////////////////////////////////////////
        /// unsafe set operation
//         tagged_thread_state& operator= (tagged_thread_state const& p)
//         {
//             state_ = p.state_;
//             return *this;
//         }

        void set (T state, tag_type t)
        {
            state_ = pack_state(state, t);
        }

        ///////////////////////////////////////////////////////////////////////
        bool operator== (volatile tagged_thread_state const & p) const
        {
            return state_ == p.state_;
        }

        bool operator!= (volatile tagged_thread_state const & p) const
        {
            return !operator==(p);
        }

        ///////////////////////////////////////////////////////////////////////
        /// state access
        T get_state() const volatile
        {
            return static_cast<T>(extract_state(state_));
        }

        void set_state(T state) volatile
        {
            tag_type t = get_tag();
            state_ = pack_state(state, t);
        }

        operator T() const volatile
        {
            return get_state();
        }

        ///////////////////////////////////////////////////////////////////////
        /// tag access
        tag_type get_tag() const volatile
        {
            return extract_tag(state_);
        }

        void set_tag(tag_type t) volatile
        {
            T state = get_state();
            state_ = pack_state(state, t);
        }

    protected:
        tagged_thread_state_type state_;
    };
}}}

#endif
