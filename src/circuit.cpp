#include "circuit.h"
      
opendp::circuit::circuit() 
: GROUP_IGNORE(false),
        num_fixed_nodes(0),
        num_cpu(1),
        DEFVersion(""),
        DEFDelimiter("/"),
        DEFBusCharacters("[]"),
        design_name(""),
        DEFdist2Microns(0),
        sum_displacement(0.0),
        displacement(400.0),
        max_disp_const(0.0),
        max_utilization(100.0),
        wsite(0),
        max_cell_height(1),
        rowHeight(0.0f), 
        fileOut(0) {

    macros.reserve(128);
    layers.reserve(32);
    rows.reserve(4096);
    sub_regions.reserve(100);

#ifdef USE_GOOGLE_HASH
    macro2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between macro name and ID */
    cell2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between cell  name and ID */
    pin2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between pin   name and ID */
    net2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between net   name and ID */
    row2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between row   name and ID */
    site2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between site  name and ID */
    layer2id.set_empty_key(
        INITSTR); /* OPENDP_HASH_MAP between layer name and ID */
    via2id.set_empty_key(INITSTR);
    group2id.set_empty_key(INITSTR); /* group between name -> index */
#endif
};

opendp::circuit::circuit(const circuit& copied){
    GROUP_IGNORE = copied.GROUP_IGNORE;
    design_util = copied.design_util;
    sum_displacement = copied.sum_displacement;
    num_fixed_nodes = copied.num_fixed_nodes;
    total_mArea = copied.total_mArea;
    total_fArea = copied.total_fArea;
    designArea = copied.designArea;
    rowHeight = copied.rowHeight;
    lx = copied.lx;
    rx  = copied.rx;
    by = copied.by;
    ty = copied.ty; 
    die = copied.die;
    core = copied.core;

   minVddCoordiY = copied.minVddCoordiY;
   initial_power = copied.initial_power;

   max_utilization = copied.max_utilization;
   displacement = copied.displacement;
   max_disp_const = copied.max_disp_const;
   wsite = copied.wsite;
   max_cell_height = copied.max_cell_height;
   num_cpu = copied.num_cpu;

   out_def_name = copied.out_def_name;
   in_def_name = copied.in_def_name;

   benchmark = copied.benchmark;

    // 2D - pixel grid;
    pixel** grid;
    grid = new pixel*[sizeof(copied.grid) / sizeof(copied.grid[0])] 
    for(int i = 0; i < sizeof(copied.grid) / sizeof(copied.grid[0]); i++){
        grid[i] = new pixel[sizeof(copied.grid[0]) / sizeof(pixel)];
    }

    for(int i = 0; i < sizeof(copied.grid) / sizeof(copied.grid[0]); i++){
        for(int j = 0; j < sizeof(copied.grid[0]) / sizeof(pixel); j++){
            grid[i][j] = copied.grid[i][j];
        }
    }

    dummy_cell = copied.dummy_cell;

    sub_regions = copied.sub_regions;
    tracks = copied.tracks;
    //얘네 그 포인터 무서워서 일단 놔둔애들
    
  // used for LEF file
    LEFVersion = copied.LEFVersion;
    LEFNamesCaseSensitive = copied.LEFNamesCaseSensitive;
    LEFDelimiter = copied.LEFDelimiter;
    LEFBusCharacters = copied.LEFBusCharacters;
    LEFManufacturingGrid = copied.LEFManufacturingGrid;

    MAXVIASTACK = copied.MAXVIASTACK;
    layer* minLayer;
    layer* maxLayer;

    DEFVersion = copied.DEFVersion;
    DEFDelimiter = copied.DEFDelimiter;
    DEFBusCharacters = copied.DEFBusCharacters;
    design_name = copied.design_name;
    DEFdist2Microns = copied.DEFdist2Microns;

    dieArea = copied.dieArea;
    sites = copied.sites;  
    layers = copied.layers; 
    macros = copied.macros; 
    cells = copied.cells;   
    nets = copied.nets; 
    pins = copied.pins;   
  
    prevrows = copied.prevrows; 
    rows = copied.rows;

    vias = copied.vias;
    viaRules = copied.viaRules;
    groups = copied.groups;

  std::vector< std::pair< double, cell* > > large_cell_stor;

#ifdef USE_GOOGLE_HASH
    macro2id = copied.macro2id;
    cell2id = copied.cell2id;
    pin2id = copied.pin2id;
    net2id = copied.net2id;
    row2id = copied.row2id;
    site2id = copied.site2id;
    layer2id = copied.layer2id;
    via2id = copied.via2id;
    group2id = copied.group2id;
#endif
};