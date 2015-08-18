//
//  Copyright (c) 2011 Bryce Lelbach
//  Copyright (c) 2011-2012 Hartmut Kaiser
//  Copyright (c) 2010 Artyom Beilis (Tonkikh)
//
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)
//
#ifndef HPX_BACKTRACE_HPP
#define HPX_BACKTRACE_HPP

#include <boost/config.hpp>
#include <string>
#include <vector>
#include <iosfwd>

///////////////////////////////////////////////////////////////////////////////
//#ifdef BOOST_HAS_DECLSPEC // defined by boost.config
// we need to import/export our code only if the user has specifically
// asked for it by defining either BOOST_ALL_DYN_LINK if they want all boost
// libraries to be dynamically linked, or HPX_BACKTRACE_DYN_LINK
// if they want just this one to be dynamically linked:
#if defined(BOOST_ALL_DYN_LINK) || defined(HPX_BACKTRACE_DYN_LINK)
// export if this is our own source, otherwise import:
#ifdef HPX_BACKTRACE_SOURCE
# define HPX_BACKTRACE_DECL BOOST_SYMBOL_EXPORT
#else
# define HPX_BACKTRACE_DECL BOOST_SYMBOL_IMPORT
#endif  // HPX_BACKTRACE_SOURCE
#endif  // DYN_LINK
//#endif  // BOOST_HAS_DECLSPEC
//
// if HPX_BACKTRACE_DECL isn't defined yet define it now:
#ifndef HPX_BACKTRACE_DECL
#define HPX_BACKTRACE_DECL
#endif

namespace hpx { namespace util
{
    namespace stack_trace
    {
        HPX_BACKTRACE_DECL std::size_t trace(void **addresses, std::size_t size);
        HPX_BACKTRACE_DECL void write_symbols(void *const *addresses,
            std::size_t size,std::ostream &);
        HPX_BACKTRACE_DECL std::string get_symbol(void *address);
        HPX_BACKTRACE_DECL std::string get_symbols(void * const *address,
            std::size_t size);
    } // stack_trace

    class backtrace {
    public:

        enum { default_stack_size = 128 };

        backtrace(std::size_t frames_no = default_stack_size)
        {
            if(frames_no == 0)
                return;
            frames_.resize(frames_no,0);
            std::size_t size = stack_trace::trace(&frames_.front(),frames_no);
            if(size != 0)
                frames_.resize(size);
        }

        virtual ~backtrace() throw()
        {
        }

        std::size_t stack_size() const
        {
            return frames_.size();
        }

        void *return_address(std::size_t frame_no) const
        {
            if(frame_no < stack_size())
                return frames_[frame_no];
            return 0;
        }

        void trace_line(std::size_t frame_no,std::ostream &out) const
        {
            if(frame_no < frames_.size())
                stack_trace::write_symbols(&frames_[frame_no],1,out);
        }

        std::string trace_line(std::size_t frame_no) const
        {
            if(frame_no < frames_.size())
                return stack_trace::get_symbol(frames_[frame_no]);
            return std::string();
        }

        std::string trace() const
        {
            if(frames_.empty())
                return std::string();
            return stack_trace::get_symbols(&frames_.front(),frames_.size());
        }

        HPX_BACKTRACE_DECL std::string trace_on_new_stack() const;

        void trace(std::ostream &out) const
        {
            if(frames_.empty())
                return;
            stack_trace::write_symbols(&frames_.front(),frames_.size(),out);
        }

    private:
        std::vector<void *> frames_;
    };

    namespace details {
        class trace_manip {
        public:
            trace_manip(backtrace const *tr) :
                tr_(tr)
            {
            }
            std::ostream &write(std::ostream &out) const
            {
                if(tr_)
                    tr_->trace(out);
                return out;
            }
        private:
            backtrace const *tr_;
        };

        inline std::ostream &operator<<(std::ostream &out,details::trace_manip const &t)
        {
            return t.write(out);
        }
    }

    template<typename E>
    inline details::trace_manip trace(E const &e)
    {
        backtrace const *tr = dynamic_cast<backtrace const *>(&e);
        return details::trace_manip(tr);
    }

    inline std::string trace(
        std::size_t frames_no = backtrace::default_stack_size) //-V659
    {
        return backtrace(frames_no).trace();
    }

    inline std::string trace_on_new_stack(
        std::size_t frames_no = backtrace::default_stack_size)
    {
        return backtrace(frames_no).trace_on_new_stack();
    }

}} // hpx::util

#endif

