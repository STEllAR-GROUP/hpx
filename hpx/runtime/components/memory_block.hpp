//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/memory_block.hpp>

namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a runtime_support class is the client side representation of a 
    /// \a server#memory_block component
    class memory_block : public stubs::memory_block
    {
    private:
        typedef stubs::memory_block base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#memory_block instance with the given global id \a gid.
        memory_block(applier::applier& app, naming::id_type gid,
                bool freeonexit = true) 
          : base_type(app), gid_(gid), freeonexit_(freeonexit)
        {
            BOOST_ASSERT(gid_);
        }

        ~memory_block() 
        {
            if (freeonexit_)
                this->base_type::free(gid_);
        }

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Get the \a memory_block_data maintained by this memory_block
        memory_block_data get(threads::thread_self& self) 
        {
            return this->base_type::get(self, gid_);
        }

        /// Asynchronously get the \a memory_block_data maintained by this 
        /// memory_block
        lcos::future_value<memory_block_data> get() 
        {
            return this->base_type::get_async(gid_);
        }

        /// Create a new instance of an memory_block on the locality as 
        /// given by the parameter \a targetgid
        static memory_block 
        create(threads::thread_self& self, applier::applier& appl, 
            naming::id_type const& targetgid, bool freeonexit = false)
        {
            return memory_block(appl, 
                base_type::create(self, appl, targetgid), freeonexit);
        }

        void free()
        {
            base_type::free(gid_);
            gid_ = naming::invalid_id;
        }

        ///////////////////////////////////////////////////////////////////////
        naming::id_type const& get_gid() const
        {
            return gid_;
        }

    private:
        naming::id_type gid_;
        bool freeonexit_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class memory_data
    {
    public:
        memory_data(memory_block_data& mb)
          : mb_(mb)
        {}

        T& operator*()
        {
            return *mb_.get<T>();
        }
        T const& operator*() const
        {
            return *mb_.get<T>();
        }

    private:
        memory_block_data mb_;
    };

}}

#endif
