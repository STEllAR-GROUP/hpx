//  Copyright (c) 2007-2010 Hartmut Kaiser
// 
//  Distributed under the Boost Software License, Version 1.0. (See accompanying 
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_MANDELBROT_CALLBACK_JANUARY_27_2009_0731PM)
#define HPX_MANDELBROT_CALLBACK_JANUARY_27_2009_0731PM

#include <hpx/hpx.hpp>

#include "mandelbrot.hpp"

namespace mandelbrot { namespace server { namespace detail
{
    ///////////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT callback
      : public hpx::lcos::base_lco_with_value<mandelbrot::result>
    {
    public:
        typedef void set_result_func_type(mandelbrot::result const&);

        callback()
        {}

        callback(boost::function<set_result_func_type> f)
          : f_(f)
        {}

        ///////////////////////////////////////////////////////////////////////
        // exposed functionality of this component
        void set_result (mandelbrot::result const& result);

        template <typename ManagedType>
        hpx::naming::id_type const& get_gid(ManagedType* p) const
        {
            if (!id_) {
                hpx::naming::gid_type gid = p->get_base_gid(); 
                hpx::naming::strip_credit_from_gid(gid);
                id_ = hpx::naming::id_type(gid, hpx::naming::id_type::unmanaged);
            }
            return id_;
        }

    private:
        boost::function<set_result_func_type> f_;
        mutable hpx::naming::id_type id_;
    };

}}}

///////////////////////////////////////////////////////////////////////////////
namespace mandelbrot { namespace server 
{
    class callback 
      : public hpx::components::managed_component<detail::callback, callback>
    {
    private:
        typedef hpx::components::managed_component<wrapped_type, callback> base_type;

    public:
        typedef detail::callback wrapped_type;

        callback()
        {}

        callback(boost::function<wrapped_type::set_result_func_type> f)
          : base_type(new base_type::wrapped_type(f))
        {}
    };

}}

#endif

