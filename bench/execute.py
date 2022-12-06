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


# run = "OnePlace"
run = "test"

ckt = opendp.circuit()
ckt_original = opendp.circuit()
ckt_original.read_files(argv)

ckt.copy_data(ckt_original)

Cell = ckt.get_Cell()

if(run == "OnePlace"):
	ckt.pre_placement()
 
	for i in range(Cell.size()):
		ckt.place_oneCell(i)
		if(i % 10 == 0):
			for j in range(Cell.size()):
				if(j > i):
					Cell[j].x_coord = Cell[j].init_x_coord
					Cell[j].y_coord = Cell[j].init_y_coord
			ckt.calc_density_factor(4)
			ckt.write_def("def/nangate45_"+str(i)+".def")
			for j in range(Cell.size()):
				if(j > i):
					Cell[j].x_coord = 0
					Cell[j].y_coord = 0
   
	ckt.calc_density_factor(4)
	ckt.write_def("def/nangate45_"+str(Cell.size())+".def")
	ckt.evaluation()
	ckt.check_legality() 
else:
	for i in range(Cell.size()):
		Cell[i].moveTry = True
	
	print(ckt.isDone_calc())
	ckt.copy_data(ckt_original)
 
	print(ckt.isDone_calc())
	
  
print(" - - - - - < Program END > - - - - - ")
