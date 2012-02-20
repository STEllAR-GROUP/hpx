

set(dependencies
    distributing_factory_component)

add_hpx_component(distributed_test
    DEPENDENCIES ${dependencies}
    FOLDER "Examples/Distributed_test"
    AUTOGLOB)
