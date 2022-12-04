%module opendp
%{
    #include "circuit.h"
%}

%include "std_string.i"
%include "std_vector.i"
%include "cpointer.i"
%include "circuit.h"

namespace std{
    %template(CellPointerVector) vector<opendp::cell*>;
}
