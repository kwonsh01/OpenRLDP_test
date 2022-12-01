import opendp

print("===========================================================================")     
print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
print("   Developers : SangGi Do, Mingyu Woo                                      ")
print("===========================================================================")

print("1: gcd_nangate45")
print("2: des_perf_a_md1")
#file = input()
file = '1'
if(file == '1'):
	argv = "opendp -lef gcd_nangate45/Nangate45_tech.lef -lef gcd_nangate45/Nangate45.lef -def gcd_nangate45/gcd_nangate45_global_place.def -cpu 4 -output_def gcd_nangate45_output.def"
	
elif(file == '2'):
	argv = "opendp -lef des_perf_a_md1/tech.lef -lef des_perf_a_md1/cells_modified.lef -def des_perf_a_md1/placed.def -cpu 4 -placement_constraints des_perf_a_md1/placement.constraints -output_def des_perf_a_md1_output.def"
	
measure = opendp.CMeasure()
ckt = opendp.circuit()

ckt.read_files(argv)

#Cell = ckt.get_Cell()
#print(Cell)

ckt.evaluation()

ckt.simple_placement(measure)
ckt.calc_density_factor(4)

ckt.write_def(ckt.out_def_name)

ckt.evaluation()

ckt.check_legality()
print(" - - - - - < Program END > - - - - - ")
