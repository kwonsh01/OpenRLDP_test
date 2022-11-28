import opendp

print("===========================================================================")     
print("   Open Source Mixed-Height Standard Cell Detail Placer < OpenDP_v1.0 >    ")
print("   Developers : SangGi Do, Mingyu Woo                                      ")
print("===========================================================================")

argv = "opendp -lef tech.lef -lef cells_modified.lef -def placed.def -cpu 4 -placement_constraints placement.constraints -output_def output.def";

measure = opendp.CMeasure()
ckt = opendp.circuit()

ckt.read_files(argv)

Cell = ckt.get_Cell()
print(Cell)

ckt.evaluation()

ckt.simple_placement(measure)
ckt.calc_density_factor(4)

ckt.write_def(ckt.out_def_name)

ckt.evaluation()

ckt.check_legality()
print(" - - - - - < Program END > - - - - - ")
