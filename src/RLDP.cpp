#include "circuit.h"
#include <vector>

using opendp::circuit;
using opendp::cell;
using opendp::row;
using opendp::pixel;
using opendp::rect;

using namespace std;

std::vector<cell*> circuit::get_Cell(){
  vector<cell*> cell_list;

  for(int i = 0; i < cells.size(); i++) {
    if(cells[i].isFixed || cells[i].inGroup || cells[i].isPlaced) continue;
    cell_list.push_back(&(cells[i]));
  }
  //sort(cell_list.begin(), cell_list.end(), SortUpOrder);
  return cell_list;
}

opendp::circuit::circuit(const circuit& copied){
    //#ifdef USE_GOOGLE_HASH
    macro2id = copied.macro2id;
    cell2id = copied.cell2id;
    pin2id = copied.pin2id;
    net2id = copied.net2id;
    row2id = copied.row2id;
    site2id = copied.site2id;
    layer2id = copied.layer2id;
    via2id = copied.via2id;
    group2id = copied.group2id;
    //#endif

    edge_spacing = copied.edge_spacing;

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
   
    grid = new pixel*[sizeof(copied.grid) / sizeof(copied.grid[0])];

    for(int i = 0; i < sizeof(copied.grid) / sizeof(copied.grid[0]); i++){
        grid[i] = new pixel[sizeof(copied.grid[0]) / sizeof(pixel)];
        
        for(int j = 0; j < sizeof(copied.grid[0]) / sizeof(pixel); j++){
            grid[i][j] = copied.grid[i][j];
        }
    }
    
    dummy_cell = copied.dummy_cell;

    sub_regions = copied.sub_regions;
    tracks = copied.tracks;
    
    LEFVersion = copied.LEFVersion;
    LEFNamesCaseSensitive = copied.LEFNamesCaseSensitive;
    LEFDelimiter = copied.LEFDelimiter;
    LEFBusCharacters = copied.LEFBusCharacters;
    LEFManufacturingGrid = copied.LEFManufacturingGrid;

    MAXVIASTACK = copied.MAXVIASTACK;

    //minLayer = new layer[sizeof(copied.minLayer)];

    //for(int i = 0; i < sizeof(copied.minLayer); i++){
    //    minLayer[i] = copied.minLayer[i];
    //}

    //maxLayer = new layer[sizeof(copied.maxLayer)];

    //for(int i = 0; i < sizeof(copied.maxLayer); i++){
    //    maxLayer[i] = copied.maxLayer[i];
    //}

    minLayer = copied.minLayer;
    maxLayer = copied.maxLayer;

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

    fileOut = copied.fileOut;   
    
    large_cell_stor = copied.large_cell_stor;

    // for(int i = 0; i < copied.large_cell_stor.size(); i++){
    //     large_cell_stor.push_back(std::pair<double, opendp::cell*>(copied.large_cell_stor[i].first, &(cells[i])));
    // };
};
