
//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <oclm/context.hpp>

#include <algorithm>

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/scoped_array.hpp>

namespace
{
    cl_context_properties * create_properties_array(std::map<int, cl_context_properties> const &props)
    {
        std::map<int, cl_context_properties>::const_iterator iter = props.begin();

        cl_context_properties * ps = new cl_context_properties[props.size() * 2 + 1];

        std::size_t i = 0;
        while(iter != props.end())
        {
            ps[i++] = iter->first;
            ps[i++] = iter->second;
            ++iter;
        }

        ps[i] = 0;

        return ps;
    }

    std::vector<oclm::context> setup_default_contexts()
    {
        std::vector<oclm::platform> const & platforms = oclm::get_platforms();
        std::vector<oclm::context> contexts;
        contexts.reserve(platforms.size());

        BOOST_FOREACH(oclm::platform const & p, platforms)
        {
            boost::scoped_array<cl_context_properties>
                props(
                    create_properties_array(
                        oclm::context_properties(
                            oclm::context_platform(p)
                        )
                    )
                );

            cl_int err = CL_SUCCESS;
            std::vector<cl_device_id> device_ids(p.devices.begin(), p.devices.end());

            // TODO: add error handling callback ...
            cl_context ctx
                = ::clCreateContext(
                    props.get()       // properties array
                  , static_cast<cl_uint>(device_ids.size()) // number of devices
                  , &device_ids[0]    // address of first device id
                  , NULL              // error notification callback
                  , NULL              // user info to be passed to the callback
                  , &err              // return error code ...
                );
            OCLM_THROW_IF_EXCEPTION(err, "clCreateContext");
            contexts.push_back(oclm::context(ctx));
        }

        return contexts;
    }
}

namespace oclm
{
    context_manager & context_manager::get()
    {
        util::static_<context_manager> man;

        return man.get();
    }

    context context_manager::default_context(device const & d)
    {
        BOOST_FOREACH(context const & ctx, default_contexts)
        {
            BOOST_FOREACH(cl_device_id device_id, ctx.get(context_devices))
            {
                if(device_id == d)
                {
                    return ctx;
                }
            }
        }

        return context();
    }

    /*
    context create_context()
    {
        return create_context(context_manager::get().default_properties);
    }

    context create_context(std::map<int, cl_context_properties> const &props)
    {
        return create_context(props, get_device());
    }

    template <cl_device_type Type>
    context create_context(device_type<Type> const & type)
    {
        return create_context(context_manager::get().default_properties, type);
    }

    context create_context(std::map<int, cl_context_properties> const & props, device const & d)
    {
        //TODO: error handling
        boost::scoped_array<cl_context_properties> ps(create_properties_array(props));
        cl_context id = ::clCreateContext(ps.get(), 1, &d.id(), NULL, NULL, NULL);
        return context(id);
    }

    template <cl_device_type Type>
    context create_context(
        std::map<int, cl_context_properties> const & props, device_type<Type> const & type)
    {
    }
    */

    context_manager::context_manager()
        : default_contexts(::setup_default_contexts())
    {
    }

    context create_context(device const & d)
    {
        //TODO: error handling
        /*
        boost::scoped_array<cl_context_properties> ps(create_properties_array(props));
        cl_context id = ::clCreateContextFromType(ps.get(), type, NULL, NULL, NULL);
        */
        return context_manager::get().default_context(d);
    }
}
