// shared_memory.hpp

// Boost Logging library
//
// Author: John Torjo, www.torjo.com
//
// Copyright (C) 2007 John Torjo (see www.torjo.com for email)
//
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE_1_0.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)
//
// See http://www.boost.org for updates, documentation, and revision history.
// See http://www.torjo.com/log2/ for more details


#ifndef JT28092007_shared_memory_HPP_DEFINED
#define JT28092007_shared_memory_HPP_DEFINED

#if defined(HPX_MSVC) && (HPX_MSVC >= 1020)
# pragma once
#endif

#if 0

#include <hpx/util/logging/detail/fwd.hpp>
#include <boost/shmem/shmem_named_shared_object.hpp>

#include <string>

namespace hpx { namespace util { namespace logging { namespace destination {

    // FIXME not tested !!!

namespace detail {
    struct shared_memory_context {
        shared_memory_context() : occupied_size(0), memory(0), mem_size(0) {
            // note: we don't want to destroy this segment, since we want
            // it to outlive our
            // application. In the case of a problem, we need to have
            // another program that will
            // take a look at what we just logged.
        }

        // how much of the memory is occupied?
        long * occupied_size;
        // the shared memory
        char_type * memory;
        hpx::util::shmem::named_shared_object segment;
        std::size_t mem_size;

    };
}

/**
    @brief Logs the information in shared memory
*/
template<class convert_dest = do_convert_destination >
struct shared_memory_t : non_const_context<detail::shared_memory_context> {

    enum { just_in_case = 8192 };
    shared_memory_t(const std::string & name, std::size_t mem_size =
        2 * 1024 * 1024 * sizeof(char_type) ) : m_name(name), m_mem_size(mem_size) {
        non_const_context_base::context().mem_size = mem_size;

        // this segment might have been previously created...
        non_const_context_base::context().segment.open_or_create(name.c_str(),
            non_const_context_base::context().mem_size + just_in_case);

        // the string
        { typedef std::pair<char_type*, std::size_t> pair;
        pair res = non_const_context_base::context().segment.find<char_type>
            ("shared_log_object");
        if ( !res.first)
            // we're creating it right now
            non_const_context_base::context().segment.construct<char_type>
            ("shared_log_object")[mem_size](0);

        res = non_const_context_base::context().segment.find<char_type>
            ("shared_log_object");
        HPX_ASSERT( res.first); // should be created by now
        non_const_context_base::context().memory = res.first;
        }

        // the occupied size
        { typedef std::pair<long*, std::size_t> pair;
        pair res = non_const_context_base::context().segment.find<long>
            ("shared_occupied_size");
        if ( !res.first)
            // we're creating it right now
            non_const_context_base::context().segment.construct<long>
            ("shared_occupied_size")[1](0);

        res = non_const_context_base::context().segment.find<long>
            ("shared_occupied_size");
        HPX_ASSERT( res.first); // should be created by now
        non_const_context_base::context().occupied_size = res.first;
        }
    }

    template<class msg_type> void operator () (const msg_type& msg_arg) const {
        const string_type & msg = do_convert::do_convert(msg_arg, into<string_type>() );

        bool can_fit = *(non_const_context_base::context().occupied_size)
            + msg.size() < non_const_context_base::context().mem_size;
        if ( can_fit) {
            std::copy(msg.begin(), msg.end(), non_const_context_base::context().memory
                + *non_const_context_base::context().occupied_size);
            *non_const_context_base::context().occupied_size += (long)msg.size();
        }
        else {
            // exceeds bounds
            if ( msg.size() < non_const_context_base::context().mem_size) {
                // move what was previously written, to the left, to make room
                std::size_t keep = non_const_context_base::context().mem_size / 2;
                if ( keep + msg.size() > non_const_context_base::context().mem_size)
                    keep = non_const_context_base::context().mem_size - msg.size();
                std::copy_backward(
                    non_const_context_base::context().memory +
                    *non_const_context_base::context().occupied_size - keep,
                    non_const_context_base::context().memory +
                    *non_const_context_base::context().occupied_size,
                    non_const_context_base::context().memory + keep);
                std::copy( msg.begin(), msg.end(),
                    non_const_context_base::context().memory + keep);
                *non_const_context_base::context().occupied_size =
                    (long)(keep + msg.size());
            }
            else {
                // message too big
                std::copy(msg.begin(), msg.begin() +
                    non_const_context_base::context().mem_size,
                    non_const_context_base::context().memory);
                *non_const_context_base::context().occupied_size =
                    (long)non_const_context_base::context().mem_size;
            }
        }
    }

    bool operator==(const shared_memory_t& other) const {
        return m_name == other.m_name && m_mem_size == other.m_mem_size;
    }

private:
    std::string m_name;
    std::size_t m_mem_size;

};

/** @brief shared_memory_t with default values. See shared_memory_t

@copydoc shared_memory_t
*/
typedef shared_memory_t<> shared_memory;

}}}}

#endif

#endif

