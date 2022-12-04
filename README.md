# OpenRLDP
*OpenDP with PPO*

NEW Function(RLDP.cpp)  
>circuit.h 
>>std::vector<cell*> get_Cell();  
>>//circuit(const circuit& copied);  
>>void copy_data(const circuit& copied);  
>>void pre_placement();  
>>void place_oneCell(int cell_id);  
>>cell* get_target_cell(int cell_id);  
>>  
>>Remove CMeasure

### How To Compile
    $ git clone --single-branch dev https://github.com/kwonsh01/OpenRLDP_test.git
    $ make

### How To Execute
    $ cd bench
    // needs to modifiy opendp.py
    $ python3 execute.py