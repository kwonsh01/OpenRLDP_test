%module opendp
%{
    #include "circuit.h"
    #include "mymeasure.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "cpointer.i"
%include "circuit.h"
%include "mymeasure.h"

namespace std{
    %template(CellPointerVector) vector<opendp::cell*>;
}
