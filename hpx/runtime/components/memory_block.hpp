//  Copyright (c) 2007-2008 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM)
#define HPX_COMPONENTS_MEMORY_BLOCK_OCT_22_2008_0416PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/runtime/applier/applier.hpp>
#include <hpx/runtime/components/stubs/memory_block.hpp>
#include <hpx/runtime/components/client_base.hpp>

#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_const.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace components 
{
    ///////////////////////////////////////////////////////////////////////////
    /// The \a runtime_support class is the client side representation of a 
    /// \a server#memory_block component
    class memory_block : public client_base<memory_block, stubs::memory_block>
    {
    private:
        typedef client_base<memory_block, stubs::memory_block> base_type;

    public:
        /// Create a client side representation for the existing
        /// \a server#memory_block instance with the given global id \a gid.
        memory_block(applier::applier& app, naming::id_type gid,
                bool freeonexit = false) 
          : base_type(app, gid, freeonexit)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component

        /// Get the \a memory_block_data maintained by this memory_block
        memory_block_data get(threads::thread_self& self) 
        {
            return this->base_type::get(self, gid_);
        }

        /// Asynchronously get the \a memory_block_data maintained by this 
        /// memory_block
        lcos::future_value<memory_block_data> get_async() 
        {
            return this->base_type::get_async(gid_);
        }
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class access_memory_block;

    template <typename T>
    class access_memory_block_proxy
    {
    private:
        typedef typename boost::remove_const<T>::type target_type;
        typedef typename 
            boost::mpl::if_<
                boost::is_const<T>, memory_block_data const&, memory_block_data&
            >::type
        wrapped_type;

    public:
        explicit access_memory_block_proxy(wrapped_type block)
          : block_(block)
        {}

        access_memory_block_proxy& operator=(target_type const& rhs)
        {
            block_.template set<target_type>(rhs);
            return *this;
        }

        operator target_type const&() const
        {
            return block_.template get<target_type>();
        }

    private:
        wrapped_type block_;
    };

    ///////////////////////////////////////////////////////////////////////////
    template <typename T>
    class access_memory_block
    {
    private:
        typedef typename boost::remove_const<T>::type target_type;

    public:
        access_memory_block()
        {}
        access_memory_block(memory_block_data const& mb)
          : mb_(mb)
        {}

        access_memory_block& operator=(memory_block_data const& mb)
        {
            mb_ = mb;
            return *this;
        }

        ///////////////////////////////////////////////////////////////////////
        access_memory_block_proxy<T> operator*()
        {
            return access_memory_block_proxy<T>(mb_);
        }
        access_memory_block_proxy<T const> operator*() const
        {
            return access_memory_block_proxy<T>(mb_);
        }

        ///////////////////////////////////////////////////////////////////////
        target_type* operator->() 
        {
            return reinterpret_cast<target_type*>(mb_.get_ptr());
        }
        target_type const* operator->() const
        {
            return reinterpret_cast<target_type const*>(mb_.get_ptr());
        }

    private:
        memory_block_data mb_;
    };

}}

#endif
