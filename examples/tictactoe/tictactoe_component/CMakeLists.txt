set(dependencies                                                                 
    distributing_factory_component
    dataflow_component)                                              
                                                                                 
add_hpx_component(tictactoe                                                        
  DEPENDENCIES ${dependencies}                                                   
  FOLDER "Examples/Tictactoe"                                                      
  AUTOGLOB)
