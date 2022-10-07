#opfmodule.so: OPFInterface.cpp 
	#g++ -c -fpic -Wall OPFInterface.cpp -std=c++17
	#g++ -shared -o opfmodule.so OPFInterface.o OPFClassifier.o 
	#rm -rf OPFInterface.o OPFClassifier.o

# all:
# 	#make vector_python_lib.so
# 	make opfmodule.so

opfmodule.so: #OPFInterface.cpp OPFClassifier.cpp
	g++ -fpic -Wall -shared OPFInterface.cpp -o opfmodule.so -Ofast -std=c++17

# vector_python_lib.so: Vector.cpp
# 	g++ -fpic -Wall -shared Vector.cpp -o vector_python_lib.so -std=c++17