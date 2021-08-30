PYTHRAN_SO=pythran_implementation.cpython-38-x86_64-linux-gnu.so
TARGETS=c_implementation cpp_implementation ${PYTHRAN_SO} c_vectorised
TSK_OBJECTS=kastore.o tskit_tables.o tskit_core.o tskit_trees.o \
	tskit_stats.o tskit_genotypes.o tskit_convert.o
CFLAGS += -g -Wall -Ikastore/c -I tskit/c -O3 -march=native
CXXFLAGS += -g -Wall -Ikastore/c -I tskit/c -O3 -march=native -std=c++14

all: $(TARGETS)

c_vectorised: c_vectorised.c libtskit.a
	${CC} ${CFLAGS} $^ -o $@ -lm

c_implementation: c_implementation.c libtskit.a
	${CC} ${CFLAGS} $^ -o $@ -lm

cpp_implementation: cpp_implementation.cc libtskit.a
	${CXX} ${CXXFLAGS} $^ -o $@ -lm

${PYTHRAN_SO}: pythran_implementation.py
	pythran pythran_implementation.py

libtskit.a: ${TSK_OBJECTS}
	${AR} rcs $@ ${TSK_OBJECTS}

kastore.o: kastore
	${CC} -c ${CFLAGS} kastore/c/kastore.c -o kastore.o

tskit_%.o: tskit
	${CC} -c ${CFLAGS} tskit/c/tskit/$*.c -o $@

kastore:
	git clone https://github.com/tskit-dev/kastore.git
	# NB!!! Make sure to checkout at a version tag!
	cd kastore && git checkout C_2.0.0

tskit:
	git clone https://github.com/tskit-dev/tskit.git
	# NB!!! Make sure to checkout at a version tag!
	cd tskit && git checkout C_0.99.13

clean:
	rm -f *.a *.o $(TARGETS)

mrproper: clean
	rm -fr tskit kastore example example.trees
