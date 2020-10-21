wget http://www.quantize.maths-fi.com/sites/default/files/quantization_grids/mult_dimensional_grids.zip -O grids.zip
wget http://www.quantize.maths-fi.com/sites/default/files/quantization_grids/one_dim_1_1000.zip -O grids_dim_1.zip
unzip -o grids.zip -d grids/ && rm grids.zip
unzip -o grids_dim_1.zip -d grids/ && rm grids_dim_1.zip
wget http://www.stat.columbia.edu/~gelman/arm/examples/police/frisk_with_noise.dat -O frisk.dat
wget http://www.stat.columbia.edu/~gelman/arm/examples/radon/radon.data -O radon.dat
