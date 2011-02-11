# just recurse into all components subdirectories and execute the scripts there
set(bh_SUDIRS 
    tlf
    itn
)


foreach(subdir ${bh_SUDIRS})
    add_subdirectory(${subdir})
endforeach()

# define list of target names for this directory
set(component_EXAMPLES 
    bh
)          

# for all targets specified above
foreach(example ${component_EXAMPLES})
    # define build target
set(${example}_client_SOURCES 
    bhc.cpp
)

set(bhc_DEPENDENCIES
    tlf_component 
    itn_component
    distributing_factory_component
)

add_hpx_executable(bhc  
    SOURCES ${${example}_client_SOURCES}
    DEPENDENCIES ${bhc_DEPENDENCIES}
)     
endforeach()
