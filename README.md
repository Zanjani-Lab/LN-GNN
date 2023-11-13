# LN-GNN
This GitHub page provides the code and data for the following work by PI's group:
"Predicting Micro/Nanoscale Colloidal Interactions through Local Neighborhood Graph Neural Networks"

-LN_GNN_binary.py:  Data generation, training, and prediction schemes for the LN-GNN framework, described as an example for the binary system of particles. The rest of the systems studied in the manuscript use the same algorithm. ‘outBinary’ includes the results of the MD simulations of the binary system of particle used for training/validation, ‘outBinary_prediction’ includes the results of MD simulations used for prediction (testing) of the ML model. Both data files were obtained from MD simulations in LAMMPS.

-Basic_IN_binary_Training.py: Data generation and training algorithm for the Basic-IN framework.

-Instance_Baesd.py: Data generation, training, and prediction schemes for the Instace-Based framework, described as an example for the binary system of particles.

-datafiles: provides link to the required datafiles for training and prediction being used by the code

