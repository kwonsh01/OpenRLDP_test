# OpenRLDP
*OpenDP with PPO*

### Required
* GCC compiler and SWIG
* Tested in Ubuntu 22.04

### How To Compile
    $ git clone --single-branch dev https://github.com/kwonsh01/OpenRLDP_test.git
    $ make

### How To Execute
    $ cd bench
    // needs to modifiy opendp.py
    $ python3 execute.py

### NEW Function(RLDP.cpp)  
* circuit.h
>std::vector<cell*> get_Cell();  
>void copy_data(const circuit& copied);  
>void pre_placement();  
>void place_oneCell(int cell_id);  
>cell* get_target_cell(int cell_id);  
>
>Removed CMeasure

### License
* OpenDP(Open Source Detailed Placement Engine) [[Link]](https://github.com/sanggido/OpenDP/tree/master)