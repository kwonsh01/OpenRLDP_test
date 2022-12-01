#include "circuit.h"
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
  vector<cell*> cell_list;

  for(int i = 0; i < cells.size(); i++) {
    if(cells[i].isFixed || cells[i].inGroup || cells[i].isPlaced) continue;
    cell_list.push_back(&(cells[i]));
  }
  //sort(cell_list.begin(), cell_list.end(), SortUpOrder);
  return cell_list;
}

opendp::circuit::circuit(const circuit& copied){
  //*this = copied;

  grid = NULL;

  int row_num = this->ty / this->rowHeight;
  int col = this->rx / this->wsite;

  grid = new pixel*[row_num];
  for(int i = 0; i < row_num; i++) {
    grid[i] = new pixel[col];
  }
  
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

void circuit::place_oneCell(int cell_id){
  //rl placement
  vector<cell*> cell_list;
    for(int i=0; i<cells.size(); i++) {
        cell* theCell = &cells[i];
        if(theCell->isPlaced || theCell->inGroup) continue;
        cell_list.push_back(theCell);
    }
    cout << "# of non-group movable cells: " << cell_list.size() << endl;
    
  cell* thecell;

  // for(int i=0; i<cells.size(); i++) {
  //     cell* theCell = &cells[i];
  //     if(theCell->isPlaced || theCell->inGroup) continue;
  //     cell_list.push_back(theCell);
  // }
  // cout << "# of non-group movable cells: " << cell_list.size() << endl;
  // bool isDone = cell_list.size() > 0 ? false : true;
  // cout << "isDone -->> " << isDone << endl;

  thecell = get_target_cell(cell_id);
  cout << "Cell id is " << thecell->id << endl;

 	if(!thecell->isPlaced){   
    if(map_move(thecell, "init_coord") == false) {
      if(shift_move(thecell, "init_coord") == false) {
        cout << thecell->name << " -> move failed!" << endl;
        //nomove = true;
		    cout << thecell->isPlaced << endl;
      }
    }
    // thecell->moveTry = true;
  }
  cout << " non_group_cell_placement done .. " << endl;
  cout << " - - - - - - - - - - - - - - - - - - - - - - - - " << endl;
  return;
}

cell* circuit::get_target_cell(int cell_id) {
  //cell* circuit::get_target_cell(Action &action) {
  //cell* get_target_cell(vector<cell*> cell_list, Action &action) {
  // Get argmax_i (cell_list)

  // sorted only in this function
  //sort(cell_list.begin(), cell_list.end(), SortByPrior);
  //cell* theCell = cell_list.front(); 

  //cell* theCell = &(cells[action->tarID]);
  cout << "target cell's ID is : " << cell_id << endl;
  cell* theCell = &(cells[cell_id]);
  cout << "target cell: " << theCell->name << endl;

  return theCell;
}

