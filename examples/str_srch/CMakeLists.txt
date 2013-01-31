
add_definitions(-DHPX_ACTION_ARGUMENT_LIMIT=10)
#add_definitions(-DHPX_COMPONENT_CREATE_ARG_MAX=10)
add_definitions(-DHPX_FUNCTION_LIMIT=13)

# add a custom target for this example                                           
add_hpx_pseudo_target(examples.str_srch.str_srch)

add_subdirectory(str_srch)

add_hpx_pseudo_dependencies(examples.str_srch examples.str_srch.str_srch)
 
source_group("Source Files" FILES str_client.cpp)

#add example executables 
add_hpx_executable(str_client
    SOURCES str_client.cpp
    DEPENDENCIES str_srch_component
    FOLDER "Examples/Str_srch")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.str_srch str_client_exe)
