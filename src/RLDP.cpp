#include "circuit.h"
#include <omp.h>
#include <vector>
#include <string>

using opendp::circuit;
using opendp::cell;
using opendp::row;
using opendp::pixel;
using opendp::rect;

using std::cout;
using std::endl;
using std::cerr;
using std::string;
using std::vector;
using std::pair;
using std::sort;
using std::make_pair;
using std::to_string;

std::vector<cell*> circuit::get_Cell(){
  for(int i = 0; i < cells.size(); i++) {
    if(cells[i].isFixed || cells[i].inGroup || cells[i].isPlaced) continue;
    cell_list_isnotFixed.push_back(&(cells[i]));
  }
  //sort(cell_list.begin(), cell_list.end(), SortUpOrder);
  return cell_list_isnotFixed;
}

void::circuit::copy_data(const circuit& copied){
  // *this = copied;//need to fix, except to copy cell_list_isnotFixed

  GROUP_IGNORE = copied.GROUP_IGNORE;

  macro2id = copied.macro2id;
  cell2id = copied.cell2id;
  pin2id = copied.pin2id;
  net2id = copied.net2id;
  row2id = copied.row2id;
  site2id = copied.site2id;
  layer2id = copied.layer2id;

  via2id = copied.via2id;
  edge_spacing = copied.edge_spacing;

  group2id = copied.group2id;

  design_util = copied.design_util;
  sum_displacement = copied.sum_displacement;

  num_fixed_nodes = copied.num_fixed_nodes;
  total_mArea = copied.total_mArea;
  total_fArea = copied.total_fArea;
  designArea = copied.designArea;
  rowHeight = copied.rowHeight;
  lx = copied.lx;
  rx = copied.rx;
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

  grid = copied.grid;
  dummy_cell = copied.dummy_cell;
  sub_regions = copied.sub_regions;
  tracks = copied.tracks;

  LEFVersion = copied.LEFVersion;
  LEFNamesCaseSensitive = copied.LEFNamesCaseSensitive;
  LEFDelimiter = copied.LEFDelimiter;
  LEFBusCharacters = copied.LEFBusCharacters;
  LEFManufacturingGrid = copied.LEFManufacturingGrid;

  MAXVIASTACK = copied.MAXVIASTACK;
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

  large_cell_stor = copied.large_cell_stor;

  reward = -99999999999;
  isDone = false;
  //cell_list_isnotFixed;

  int row_num = this->ty / this->rowHeight;
  int col = this->rx / this->wsite;
/*
  grid = NULL;
  grid = new pixel*[row_num];
  for(int i = 0; i < row_num; i++) {
    grid[i] = new pixel[col];
  }
*/
  for(int i = 0; i < row_num; i++) {
    for(int j = 0; j < col; j++) {
      this->grid[i][j].name = "pixel_" + to_string(i) + "_" + to_string(j);
      this->grid[i][j].y_pos = i;
      this->grid[i][j].x_pos = j;
      this->grid[i][j].linked_cell = NULL;
      this->grid[i][j].isValid = false;
    }
  }
  for(auto& curFragRow : prevrows) {
    int x_start = IntConvert((1.0*curFragRow.origX - core.xLL) / wsite);
    int y_start = IntConvert((1.0*curFragRow.origY - core.yLL) / rowHeight);
    
    int x_end = x_start + curFragRow.numSites;
    int y_end = y_start + 1;

    for(int i=x_start; i<x_end; i++) {
      for(int j=y_start; j<y_end; j++) {
        grid[j][i].isValid = true;
      }
    }
  }
  
  fixed_cell_assign();
  group_pixel_assign_2();
  group_pixel_assign();
  init_large_cell_stor();

}

void circuit::pre_placement() {
  if(groups.size() > 0) {
    // group_cell -> region assign
    group_cell_region_assign();
    cout << " group_cell_region_assign done .." << endl;
  }
  // non group cell -> sub region gen & assign
  non_group_cell_region_assign();
  cout << " non_group_cell_region_assign done .." << endl;
  cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;

  // pre placement out border ( Need region assign function previously )
  if(groups.size() > 0) {
    group_cell_pre_placement();
    cout << " group_cell_pre_placement done .." << endl;
    non_group_cell_pre_placement();
    cout << " non_group_cell_pre_placement done .." << endl;
    cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  }

  // naive method placement ( Multi -> single )
  if(groups.size() > 0) {
    group_cell_placement("init_coord");
    cout << " group_cell_placement done .. " << endl;
    for(int i = 0; i < groups.size(); i++) {
      group* theGroup = &groups[i];
      for(int j = 0; j < 3; j++) {
        int count_a = group_refine(theGroup);
        int count_b = group_annealing(theGroup);
        if(count_a < 10 || count_b < 100) break;
      }
    }
  }
}

void circuit::place_oneCell(int cell_idx){
  cell* thecell;
  thecell = cell_list_isnotFixed[cell_idx];
  //cout << "Cell id is " << thecell->id << endl;
  
  if(!thecell->isPlaced == true){
    if(map_move(thecell, "init_coord") == false) {
      if(shift_move(thecell, "init_coord") == false) {
        cout << thecell->name << " -> move failed!" << endl;
        //nomove = true;
		    cout << thecell->isPlaced << endl;
      }
    }
    thecell->moveTry = true;
  }
    
  //feature update
  thecell->disp = abs(thecell->init_x_coord - thecell->x_coord) + abs(thecell->init_y_coord - thecell->y_coord);
  
  cout << cell_idx << "'s cell_placement done .. " << endl;
  // cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  return;
}

cell* circuit::get_target_cell(int cell_idx) {
  //need to improve to log n 
  cell* theCell = (cell_list_isnotFixed[cell_idx]);
  cout << "target cell's Index is : " << cell_idx << endl;
  cout << "target cell's ID is : " << theCell->id << endl;
  cout << "target cell: " << theCell->name << endl;
  return theCell;
}

bool circuit::isDone_calc() {//need to fix
  for(int i = 0; i < cell_list_isnotFixed.size(); i++){
    if(!(cell_list_isnotFixed[i]->moveTry))
      return false;
  }
  return true;
} 

double circuit::reward_calc_test() {
  double avg_disp = 0;
  double max_disp = 0;
  double hpwl = 0;
  double displacement = 0;
  int count_displacement = 0;
  int tot_ov_num = 0;

  //calc overlap number: can replace Rtree method
  for(int i = 0; i < cells.size(); i++){
    cell* theCell = &cells[i];
    int ov_num = 0;
    /*
    if (!theCell->isPlaced){
        double lx = theCell->x_coord;
        double hx = theCell->x_coord + theCell->width;
        double ly = theCell->y_coord;
        double hy = theCell->y_coord + theCell->height;

      ov_num += overlap_num_calc(theCell)
      for (int j = 0; j <cells.size(); j++){
        cell* Cell = &cells[j];
        double lx1 = Cell->x_coord;
        double hx1 = Cell->x_coord+Cell->width;
        double ly1 = Cell->y_coord;
        double hy1 = Cell->y_coord + Cell->height;
        
        if((lx < hx1 && hx1 < hx) || (lx < lx1 && lx1 < hx)){
            if( (ly < hy1 && hy1 < hy) || (ly < ly1 && ly1 < hy)){
              ov_num++;
          tot_ov_num++;
            }
        }
      }
    }
    */
    theCell->overlapNum = ov_num;
    //cout << "In reward calc : " << theCell->disp << endl;
    if(theCell->moveTry){
      displacement = abs(theCell->init_x_coord - theCell->x_coord) + abs(theCell->init_y_coord - theCell->y_coord);
      avg_disp += displacement;
      if(displacement > max_disp){
          max_disp = displacement;
      }
      count_displacement++;
    }
  }

  avg_disp = avg_disp / count_displacement;

  double shpwl = std::max((HPWL("CUR") - HPWL("INIT")) / HPWL("INIT"), 0.0); 

  // return (100 / (1 + avg_disp)) + (5 / (1 + max_disp)) + (100 / (1 + shpwl));
  return (100 / (1 + avg_disp));
}

double circuit::reward_calc() {
  //Disp calc start
  double avg_displacement = 0;
  double sum_displacement = 0;
  double max_displacement = 0;
  int count_displacement = 0;
  double violated_const = 0;
  
  int H1_count = 0;
  int H2_count = 0;
  int H3_count = 0;
  int H4_count = 0;

  int exist_H1 = 0;
  int exist_H2 = 0;
  int exist_H3 = 0;
  int exist_H4 = 0;

  double disp_H1 = 0;
  double disp_H2 = 0;
  double disp_H3 = 0;
  double disp_H4 = 0;    

  int tot_ov_num = 0;
  for(int i = 0; i < cells.size(); i++){
    cell* theCell = &cells[i];
    int ov_num = 0;
    if (!theCell->isPlaced){
        double lx = theCell->x_coord;
        double hx = theCell->x_coord + theCell->width;
        double ly = theCell->y_coord;
        double hy = theCell->y_coord + theCell->height;

      //ov_num += overlap_num_calc(theCell)
      for (int j = 0; j <cells.size(); j++){
        cell* Cell = &cells[j];
        double lx1 = Cell->x_coord;
        double hx1 = Cell->x_coord+Cell->width;
        double ly1 = Cell->y_coord;
        double hy1 = Cell->y_coord + Cell->height;
        
        if((lx < hx1 && hx1 < hx) || (lx < lx1 && lx1 < hx)){
            if( (ly < hy1 && hy1 < hy) || (ly < ly1 && ly1 < hy)){
              ov_num++;
          tot_ov_num++;
            }
        }
      }
    }

    theCell->overlapNum = ov_num;
    //cout << "In reward calc : " << theCell->disp << endl;

    double displacement = abs(theCell->init_x_coord - theCell->x_coord) + abs(theCell->init_y_coord - theCell->y_coord);
    sum_displacement += displacement;
    if(displacement > max_displacement){
        max_displacement = displacement;
    }
    count_displacement++;

    if((displacement/rowHeight) > max_disp_const) 
      violated_const += static_cast<double>(displacement/rowHeight);
    double rowheight = 2800;
    if(theCell->height == rowheight){
        H1_count++;
        disp_H1 += displacement;
        exist_H1 = 1;
    }
    if(theCell->height == 2*rowheight){
        H2_count++;
        disp_H2 += displacement;
        exist_H2 = 1;
    }
    if(theCell->height == 3*rowheight){
        H3_count++;
        disp_H3 += displacement;
        exist_H3 = 1;
    }
    if(theCell->height == 4*rowheight){
        H4_count++;
        disp_H4 += displacement;
        exist_H4 = 1;
    }
  }
  
  avg_displacement = sum_displacement / count_displacement;

  //Smm calc start
  double Smm;
  if((violated_const/max_disp_const)>1)
      Smm = 1 + static_cast<double>((violated_const/max_disp_const)*(max_displacement/(100*rowHeight)));
  else
      Smm = 1 + static_cast<double>(max_displacement/(100*rowHeight));

  //Sam calc start
  double Sam;
  double avg_disp_H1 = 0;
  double avg_disp_H2 = 0;
  double avg_disp_H3 = 0;
  double avg_disp_H4 = 0;
  if(exist_H1) avg_disp_H1 = static_cast<double>(disp_H1/H1_count);
  if(exist_H2) avg_disp_H2 = static_cast<double>(disp_H2/H2_count);
  if(exist_H3) avg_disp_H3 = static_cast<double>(disp_H3/H3_count);
  if(exist_H4) avg_disp_H4 = static_cast<double>(disp_H4/H4_count);

  Sam = static_cast<double>((avg_disp_H1 + avg_disp_H2 + avg_disp_H3 + avg_disp_H4)/(exist_H1 + exist_H2 + exist_H3 + exist_H4));
  //cout << "Here" << endl;    

  double shpwl = std::max((HPWL("CUR") - HPWL("INIT")) / HPWL("INIT"), 0.0) * (1.2);  
  // double shpwl = std::max((HPWL("CUR") - HPWL("INIT") / HPWL("INIT")), 0.0) * (1 + std::max(calc_density_factor(8.0), 0.2));

  // cout << " AVG_displacement : " << avg_displacement << endl;
  // cout << " SUM_displacement : " << sum_displacement << endl;
  // cout << " MAX_displacement : " << max_displacement << endl;
  // cout << " Smm              : " << Smm << endl;
  // cout << " Sam              : " << Sam/rowHeight << endl;
  // cout << " Shpwl            : " << shpwl << endl;
  // cout << " HPWL             : " << HPWL("CUR") << "    " << HPWL("INIT") << endl;
  double S_total = Sam*(1+shpwl)/rowHeight; //+ tot_ov_num;
  //cout << "total overlap is " << tot_ov_num << endl;
  // cout << "Stotal is " << S_total << endl;
  return S_total;
}

int circuit::overlap_num_calc(cell* theCell) {
    vector<cell*> ovcells = overlap_cells(theCell);
    ovcells.erase(unique(ovcells.begin(), ovcells.end()), ovcells.end());
    int num = (int)ovcells.size();
    /////////// num = 0 for all cells...

    if(num != 0)
        cout << "CELL " << theCell->name << " has " << num << "overlap cells" << endl;

    return num;
}