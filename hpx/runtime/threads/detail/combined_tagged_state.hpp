//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TAGGED_THREAD_STATE_MAR_12_2010_0125PM)
#define HPX_TAGGED_THREAD_STATE_MAR_12_2010_0125PM

#include <hpx/config.hpp>
#include <hpx/util/assert.hpp>

#include <cstddef>
#include <cstdint>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace threads { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T1, typename T2>
    class combined_tagged_state
    {
    private:
        typedef std::int64_t tagged_state_type;

        typedef std::int8_t thread_state_type;
        typedef std::int8_t thread_state_ex_type;
        typedef std::int64_t tag_type;

        static const std::size_t state_shift    = 56;  // 8th byte
        static const std::size_t state_ex_shift = 48;  // 7th byte

        static const tagged_state_type state_mask = 0xffull;
        static const tagged_state_type state_ex_mask = 0xffull;

        // (1L << 48L) - 1;
        static const tagged_state_type tag_mask = 0x0000ffffffffffffull;

        static tag_type extract_tag(tagged_state_type const& i)
        {
            return i & tag_mask;
        }

        static thread_state_type extract_state(tagged_state_type const& i)
        {
            return (i >> state_shift) & state_mask;
        }

        static thread_state_ex_type extract_state_ex(tagged_state_type const& i)
        {
            return (i >> state_ex_shift) & state_ex_mask;
        }

        static tagged_state_type pack_state(tagged_state_type state,
            tagged_state_type state_ex, tag_type tag)
        {
            HPX_ASSERT(!(state & ~state_mask));
            HPX_ASSERT(!(state_ex & ~state_ex_mask));
            HPX_ASSERT(!(state & ~tag_mask));

            return (state << state_shift) | (state_ex << state_ex_shift) | tag;
        }

    public:
        ///////////////////////////////////////////////////////////////////////
        combined_tagged_state()
          : state_(0)
        {}

        combined_tagged_state(T1 state, T2 state_ex, tag_type t = 0)
          : state_(pack_state(state, state_ex, t))
        {}

        combined_tagged_state(combined_tagged_state state, tag_type t)
          : state_(pack_state(state.state(), state.state_ex(), t))
        {}

        ///////////////////////////////////////////////////////////////////////
        void set (T1 state, T2 state_ex, tag_type t)
        {
            state_ = pack_state(state, state_ex, t);
        }

        ///////////////////////////////////////////////////////////////////////
        bool operator== (combined_tagged_state const & p) const
        {
            return state_ == p.state_;
        }

        bool operator!= (combined_tagged_state const & p) const
        {
            return !operator==(p);
        }

        ///////////////////////////////////////////////////////////////////////
        // state access
        T1 state() const
        {
            return static_cast<T1>(extract_state(state_));
        }

        void set_state(T1 state)
        {
            state_ = pack_state(state, state_ex(), tag());
        }

        T2 state_ex() const
        {
            return static_cast<T2>(extract_state_ex(state_));
        }

        void set_state_ex(T2 state_ex)
        {
            state_ = pack_state(state(), state_ex, tag());
        }

        ///////////////////////////////////////////////////////////////////////
        // tag access
        tag_type tag() const
        {
            return extract_tag(state_);
        }

        void set_tag(tag_type t)
        {
            state_ = pack_state(state(), state_ex(), t);
        }

    protected:
        tagged_state_type state_;
    };
}}}

#endif
