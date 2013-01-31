
# add a custom target for this example
add_hpx_pseudo_target(examples.tictactoe.tictactoe)

add_subdirectory(tictactoe_component)

add_hpx_pseudo_dependencies(examples.tictactoe examples.tictactoe.tictactoe)

source_group("Source Files" FILES tictactoe_client.cpp)

#add example executables
add_hpx_executable(tictactoe_client
    SOURCES tictactoe_client.cpp
    DEPENDENCIES tictactoe_component
    FOLDER "Examples/Tictactoe")

# add dependencies to pseudo-target
add_hpx_pseudo_dependencies(examples.tictactoe tictactoe_client_exe)
