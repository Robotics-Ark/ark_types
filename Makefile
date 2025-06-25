all:
	lcm-gen -p ark_type_defs/*
	pip install -e .
clean:
	rm -rf arktypes
	pip uninstall --yes arkypes