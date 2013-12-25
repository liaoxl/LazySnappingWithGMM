#
# Makefile
# moondark, 2013-12-18 20:13
#

CC=g++
CFLAGS=-g -w `pkg-config opencv --libs --cflags opencv`

all:\
	LazySnapping

LazySnapping: \
	maxflow-v3.01/graph.cpp \
	maxflow-v3.01/maxflow.cpp \
	main.cpp
	$(CC) $(CFLAGS) $^ -o $@

clean:
	rm LazySnapping


# vim:ft=make
#

