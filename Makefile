all:
	lcm-gen -p ark_type_defs/*
clean:
	rm -rf arktypes
	pip uninstall --yes arkypes