make -> create _opendp.so in /bench

NEW Function(RLDP.cpp)  
>circuit.h 
>>std::vector<cell*> get_Cell();  
>>//circuit(const circuit& copied);  
>>void copy_data(const circuit& copied);  
>>void pre_placement();  
>>void place_oneCell(int cell_id);  
>>cell* get_target_cell(int cell_id);  
