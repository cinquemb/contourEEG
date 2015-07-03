CFLAGS= -Wall -g -DNDEBUG -Os -std=c++11 -pedantic `pkg-config --cflags --libs librsvg-2.0` `Magick++-config --cppflags --cxxflags --ldflags --libs` `pkg-config --cflags --libs opencv` `pkg-config --cflags --libs magics` `pkg-config --cflags --libs jsoncpp` `pkg-config --cflags --libs eigen3`

all:
	ccache clang++ spacial_svg_generation.cpp -o spacial_svg_generation $(CFLAGS)

clean:
	rm spacial_svg_generation

debug:
	ccache clang++ -g spacial_svg_generation.cpp -o spacial_svg_generation $(CFLAGS)
