import numpy as np
import opendp

print("===========================================================================")     
print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
print("   Developers : SangGi Do, Mingyu Woo                                      ")
print("   RL : SH kwon, SH Kim, CH Lee                                            ")
print("===========================================================================")

# print("1: gcd_nangate45")
# print("2: des_perf_a_md1")
#file = input()
file = '1'
if(file == '1'):
	argv = "opendp -lef gcd_nangate45/Nangate45_tech.lef -lef gcd_nangate45/Nangate45.lef -def gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
elif(file == '2'):
	argv = "opendp -lef des_perf_a_md1/tech.lef -lef des_perf_a_md1/cells_modified.lef -def des_perf_a_md1/placed.def -cpu 4 -placement_constraints des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"


run = "OnePlace"

ckt = opendp.circuit()
ckt_original = opendp.circuit()
ckt_original.read_files(argv)
ckt.copy_data(ckt_original)

Cell = ckt.get_Cell()

# state = []

# for j in range(Cell.size()):
# 	disp_temp = Cell[j].disp
# 	state.append(disp_temp)

# state = np.array(state)
# print(state)

if(run == "OnePlace"):
	ckt.pre_placement()
 
	for i in range(Cell.size()):
     
		ckt.place_oneCell(Cell[i].id)
		# state[i].disp = abs(Cell[i].init_x_coord - Cell[i].x_coord) + abs(Cell[i].init_y_coord - Cell[i].y_coord)
		# state[i] = Cell[i].disp
  
		if(i % 10 == 0):
			ckt.write_def("def/gcd_nangate45_output_"+str(i))
   
	ckt.calc_density_factor(4)
	ckt.write_def(ckt.out_def_name)
	ckt.evaluation()
	ckt.check_legality() 
else:
	ckt.simple_placement()
	ckt.calc_density_factor(4)
	#ckt.write_def(ckt.out_def_name)
	ckt.evaluation()
	ckt.check_legality()
print(" - - - - - < Program END > - - - - - ")
