
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef OCLM_CONTEXT_HPP
#define OCLM_CONTEXT_HPP

#include <vector>

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/stringize.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/tuple/elem.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/repetition/enum_binary_params.hpp>
#include <boost/preprocessor/repetition/repeat_from_to.hpp>

#include <boost/mpl/bool.hpp>

#include <vector>

#include <oclm/context_properties.hpp>
#include <oclm/context_info.hpp>
#include <oclm/platform.hpp>
#include <oclm/device.hpp>

#include <CL/cl.h>

namespace oclm
{
    // TODO: add proper caching of contexts and support for device lists
    struct context
    {
        context()
            : id_(0)
        {
        }

        context(cl_context id)
            : id_(id)
        {
            if(id_ != 0) ::clRetainContext(id_);
        }

        context(context const & ctx)
            : id_(ctx.id_)
        {
            if(id_ != 0) ::clRetainContext(id_);
        }

        context & operator=(context const & ctx)
        {
            if(id_ != ctx.id_)
            {
                id_ = ctx.id_;
                if(id_ != 0) ::clRetainContext(id_);
            }

            return *this;
        }

        ~context()
        {
            if(id_ != 0) ::clReleaseContext(id_);
        }

        template <typename Info>
        typename boost::enable_if<
            typename is_context_info<Info>::type
          , typename Info::result_type
        >::type
        get(Info) const
        {
            return get_info<Info>(id_);
        }

        template <typename Info>
        typename boost::disable_if<
            typename is_context_info<Info>::type
          , void
        >::type
        get(Info) const
        {
            static_assert(
                is_context_info<Info>::value
              , "Template parameter is not a valid context info type"
            );
        }

        operator cl_context const & () const
        {
            return id_;
        }

        private:
            cl_context id_;
    };

    struct context_manager
    {
        typedef std::map<int, cl_context_properties> context_properties_type;

        context_manager();

        context default_context(device const &);

        static context_manager & get();

        std::vector<context> default_contexts;
    };

    /*
    context create_context();

    context create_context(std::map<int, cl_context_properties> const &);

    context create_context(std::map<int, cl_context_properties> const &, device const & d);

    template <cl_device_type Type>
    context create_context(
        std::map<int, cl_context_properties> const & props, device_type<Type> const & type);

    template <cl_device_type Type>
    context create_context(device_type<Type> const & type);
    */

    context create_context(device const & d);

}

#endif
