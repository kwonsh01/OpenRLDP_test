import logging
import numpy as np
import opendp

logging.basicConfig(filename='dummy.log', level=logging.INFO)

print("===========================================================================")     
print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
print("   Developers : SangGi Do, Mingyu Woo                                      ")
print("   RL : SH kwon, SH Kim, CH Lee                                            ")
print("===========================================================================")


file = 'nangate45'
if(file == 'nangate45'):
	argv = "opendp -lef benchmarks/gcd_nangate45/Nangate45_tech.lef -lef benchmarks/gcd_nangate45/Nangate45.lef -def benchmarks/gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
elif(file == 'des_perf_a_md1'):
	argv = "opendp -lef benchmarks/des_perf_a_md1/tech.lef -lef benchmarks/des_perf_a_md1/cells_modified.lef -def benchmarks/des_perf_a_md1/placed.def -cpu 4 -placement_constraints benchmarks/des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"
elif(file == 'des_perf_1'):
	argv = "opendp -lef benchmarks/des_perf_1/tech.lef -lef benchmarks/des_perf_1/cells_modified.lef -def benchmarks/des_perf_1/placed.def -cpu 4 -placement_constraints benchmarks/des_perf_1/placement.constraints -output_def des_perf_1_output.def"
elif(file == 'fft_2_md2'):
	argv = "opendp -lef benchmarks/fft_2_md2/tech.lef -lef benchmarks/fft_2_md2/cells_modified.lef -def benchmarks/fft_2_md2/placed.def -cpu 4 -placement_constraints benchmarks/fft_2_md2/placement.constraints -output_def fft_2_md2_output.def"
elif(file == 'fft_a_md2'):
	argv = "opendp -lef benchmarks/fft_a_md2/tech.lef -lef benchmarks/fft_a_md2/cells_modified.lef -def benchmarks/fft_a_md2/placed.def -cpu 4 -placement_constraints benchmarks/fft_a_md2/placement.constraints -output_def fft_a_md2_output.def"
elif(file == 'fft_a_md3'):
	argv = "opendp -lef fft_a_md2/tech.lef -lef benchmarks/fft_a_md3/cells_modified.lef -def benchmarks/fft_a_md3/placed.def -cpu 4 -placement_constraints benchmarks/fft_a_md3/placement.constraints -output_def fft_a_md3_output.def" 
elif(file == 'pci_bridge32_b_md2'):
	argv = "opendp -lef benchmarks/pci_bridge32_b_md2/tech.lef -lef benchmarks/pci_bridge32_b_md2/cells_modified.lef -def pci_bridge32_b_md2/placed.def -cpu 4 -placement_constraints pci_bridge32_b_md2/placement.constraints -output_def pci_bridge32_b_md2_out.def"
     
ckt = opendp.circuit()
ckt_original = opendp.circuit()
ckt.read_files(argv)

ckt_original.copy_data(ckt)

Cell = ckt.get_Cell()

if(1):
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
		i = Cell.size() - i - 1
		ckt.place_oneCell(i)
	ckt.calc_density_factor(4)
	ckt.evaluation()
	ckt.check_legality()
	ckt.write_def("output/sival.def")
	
print(" - - - - - < Program END > - - - - - ")
