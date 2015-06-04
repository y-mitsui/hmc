CFLAGS=-Wall -O0 -g -DHAVE_INLINE -I/usr/include/atlas/ `gsl-config --cflags` `pkg-config --cflags glib-2.0` `pkg-config --cflags apr-1` `xml2-config --cflags` `pkg-config --cflags dpgmm`
ASFLAGS=-g `pkg-config --libs standard` `pkg-config --libs glib-2.0` `pkg-config --libs apr-1` `xml2-config --libs` 
CC=gcc
LOADLIBES=-lpthread -lm -L/usr/lib -lgsl  -lm -latlas -lcblas
pmcmc:mnorm.o
clean:
	rm *.o
