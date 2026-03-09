
Cases are stored according to their design parameters. A
hash is generated using the relevant parameters and then
the directory is named using the hash. A database (either
SQL or a plain .csv file) will match case to folder.

└── #########
    └── results

This folder will house the solar field layout and flux images
for each heliostat on the receiver. The results folder holds
the products of an optimization, such as the aimpoint strategy.

These directories will be automatically created and
populated according to design parameters from inputs.py.

