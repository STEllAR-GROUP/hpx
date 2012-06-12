
#include <oclm/command_queue.hpp>
#include <oclm/context.hpp>

namespace oclm
{
    /*
    command_queue::command_queue(device const & d)
        : ctx_(create_context(d))
        , d_(d)
    {
        context ctx = create_context(d);
        cl_int err = CL_SUCCESS;
        cq
            = ::clCreateCommandQueue(
                ctx
              , d
              , CL_QUEUE_PROFILING_ENABLE
              , &err
            );

        OCLM_THROW_IF_EXCEPTION(err, "clCreateCommandQueue");
    }
    */
}
