TARGET = _opendp.so
OBJECTS = opendp_wrap.o
SOURCE = src

$(TARGET) : $(OBJECTS)
	g++ -shared -o $(TARGET) *.o -L$(CURDIR)/$(SOURCE)/lib -lcdef -lcdefzlib -ldef -ldefzlib -llef -llefzlib -lclef -lclefzlib
	mv $(TARGET) bench/
	rm *.o
	rm opendp_wrap.cxx

opendp_wrap.o :
	cp $(SOURCE)/circuit.h $(CURDIR)/
	swig -c++ -python opendp.i
	g++ -fPIC -c $(SOURCE)/*.cpp -std=c++11
	g++ -fPIC -c opendp_wrap.cxx -I/usr/include/python3.8/ -std=c++11
	#g++ -fPIC -c $(SOURCE)/opendp_wrap.cxx -I/usr/include/python3.10/
	mv opendp.py bench/
	mv circuit.h $(SOURCE)/

clean :
	rm -rf bench/$(TARGET) bench/opendp.py opendp_wrap.o *.o opendp_wrap.cxx
	rm -rf bench/__pycache__
	rm -rf bench/def
	mkdir bench/def