import logging
import numpy as np
import opendp

logging.basicConfig(filename='dummy.log', level=logging.INFO)

print("===========================================================================")     
print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
print("   Developers : SangGi Do, Mingyu Woo                                      ")
print("   RL : SH kwon, SH Kim, CH Lee                                            ")
print("===========================================================================")


file = 'pci_bridge32_b_md2'
# run = "OnePlace"
run = "test"


if(file == 'gcd_nangate45'):
	argv = "opendp -lef gcd_nangate45/Nangate45_tech.lef -lef gcd_nangate45/Nangate45.lef -def gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
elif(file == 'des_perf_a_md1'):
	argv = "opendp -lef des_perf_a_md1/tech.lef -lef des_perf_a_md1/cells_modified.lef -def des_perf_a_md1/placed.def -cpu 4 -placement_constraints des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"
elif(file == 'des_perf_1'):
	argv = "opendp -lef des_perf_1/tech.lef -lef des_perf_1/cells_modified.lef -def des_perf_1/placed.def -cpu 4 -placement_constraints des_perf_1/placement.constraints -output_def des_perf_1_output.def"
elif(file == 'pci_bridge32_b_md2'):
    argv = "opendp -lef pci_bridge32_b_md2/tech.lef -lef pci_bridge32_b_md2/cells_modified.lef -def pci_bridge32_b_md2/placed.def -cpu 4 -placement_constraints pci_bridge32_b_md2/placement.constraints -output_def pci_bridge32_b_md2_out.def"

ckt = opendp.circuit()
ckt_original = opendp.circuit()
ckt.read_files(argv)

ckt_original.copy_data(ckt)

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
			ckt.write_def("output/"+file+"_"+str(i)+".def")
			for j in range(Cell.size()):
				if(j > i):
					Cell[j].x_coord = 0
					Cell[j].y_coord = 0
   
	ckt.calc_density_factor(4)
	ckt.write_def("output/"+file+"_"+str(Cell.size())+".def")
	ckt.evaluation()
	ckt.check_legality() 
else:
	ckt.pre_placement()
	for i in range(Cell.size()):
		#logging.info(str(i))
		ckt.place_oneCell(i)
	ckt.calc_density_factor(4)
	ckt.evaluation()
	ckt.check_legality()
	ckt.write_def("output/sival.def")
	
print(" - - - - - < Program END > - - - - - ")
