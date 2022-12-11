# OpenRLDP
*OpenDP with PPO by SH Kwon, SH Kim, CH Lee*

![Random DP](https://user-images.githubusercontent.com/88278994/206904712-639e90f2-1295-47e4-bac4-a54263cdac63.gif)

### Required
* GCC compiler and SWIG
* Tested in Ubuntu 20.04

### How To Compile
    $ git clone https://github.com/kwonsh01/OpenRLDP_test.git
    $ make

### How To Execute
    $ cd bench
    $ python3 PPO.py
    $ cd data
    $ python3 mean.py

### NEW Feature(RLDP.cpp / circuit.h)  
* Class circuit  
>double reward  
>bool isDone  
>std::vector<cell*> cell_list_isnotFixed  
>std::vector<cell*> get_Cell()  
>void pre_placement()  
>void place_oneCell(int cell_idx)  
>cell* get_target_cell(int cell_idx)  
>void copy_data(const circuit& copied)  
>bool isDone_calc()  
>double reward_calc()  
>int overlap_num_calc(cell* theCell)  
>void overlap_num_calc()  
* Class cell
>int overlapNum  
>bool moveTry  
>int localOverlap  
>double localUtil  
>double hpwl  
>double prior    
* Removed CMeasure

### Reference
* OpenDP(Open Source Detailed Placement Engine) [[Link]](https://github.com/sanggido/OpenDP/tree/master)
