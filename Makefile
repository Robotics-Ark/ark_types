all:
	pip install lcm # ensure lcm installed in order to call lcm-gen
	lcm-gen -p ark_type_defs/*
clean:
	rm -rfv _arktypes_*
