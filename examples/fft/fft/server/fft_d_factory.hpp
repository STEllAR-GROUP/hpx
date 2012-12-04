#if !defined(HPX_CNjQi7mJW6Xtvbl68qCmhKKv6NWorcSxYWRfEYJb)
#define HPX_CNjQi7mJW6Xtvbl68qCmhKKv6NWorcSxYWRfEYJb
#include <hpx/hpx_fwd.hpp>                                                       
#include <hpx/runtime/components/component_type.hpp>                             
#include <hpx/runtime/components/server/simple_component_base.hpp>               
#include <hpx/runtime/actions/component_action.hpp>                              
#include <hpx/util/locality_result.hpp>                                          

#include <boost/serialization/serialization.hpp>                                 
#include <boost/serialization/vector.hpp>                                        
#include <boost/foreach.hpp>                                                     

#include <vector>    

//////////////////////////////////////////////////////////////////////////////

namespace fft {namespace server 
{
    /////////////////////////////////////////////////////////////////////////
    class HPX_COMPONENT_EXPORT d_factory
        : public hpx::components::simple_component_base<d_factory>
    {
    public:
        typedef std::vector<hpx::util::remote_locality_result> remote_result_type;
        typedef std::vector<hpx::util::locality_result> result_type;

        //typedef hpx::lcos::future<result_type, remote_result_type> 
        typedef hpx::lcos::future<result_type> 
            async_create_result_type;

        enum actions
        {
            d_factory_create_components = 0
        };
        // Create Components with cardinality of localities into perspective
        // basically for structured data distribution
        remote_result_type create_components_structured(hpx::components::component_type
            type, std::size_t count) const;

        //HPX_DEFINE_COMPONENT_CONST_ACTION(d_factory, create_components_structured
        //    , create_components_structured_action);
        typedef hpx::actions::result_action2<
            d_factory const, remote_result_type
            , hpx::components::component_type, std::size_t
            , &d_factory::create_components_structured
        > create_components_structured_action;
       
    };
}}

///////////////////////////////////////////////////////////////////////////////  
// Declaration of serialization support for the distributing_factory actions     
HPX_REGISTER_ACTION_DECLARATION(                                              
    fft::server::d_factory::create_components_structured_action      
    , d_factory_create_components_structured_action
    )

#endif //HPX_CNjQi7mJW6Xtvbl68qCmhKKv6NWorcSxYWRfEYJb
