# Neural Netowrk for Temporal Evolution of Frost Thickness

## Getting Started

### Prerequisites

Install Python and TensorFlow

* [Anaconda](https://www.anaconda.com/products/individual)
* [TensorFlow](https://www.tensorflow.org/install)

### Using the neural networks
Clone or download the [repository](https://github.com/shahaneshantanu/frost-thickness-neural-network)

## Running the Code
Currently, neural networks have been trained using the experimental data of temporal evolution of frost thickness. Raw experimental data can be accessed from this [file](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/Neural%20Networks/Raw%20Data.xlsx)

Frost thickness is predicted as a function of:

1. Surface type: defined in codes ([train_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/train_NN_script.py) and [predict_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/predict_NN_script.py)) by variable surface_type. Possible values: 'SHL', 'SHP' and 'R' which denote super-hydrophilic, super-hydrophobic and regular surfaces respectively
2. Seperation between the plates in mm defined in codes ([train_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/train_NN_script.py) and [predict_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/predict_NN_script.py)) by variable separation. Possible values: 2, 4, 6, 8
3. Surface temperature in Celcius defined in codes ([train_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/train_NN_script.py) and [predict_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/predict_NN_script.py)) by variable surface_temperature. Possible values: -5, -10, -15

Note that the above three variables have only finite number of possibilities due to the limited amount of experimental data currently available. But this can be expanded if additional data is obtained in future.

### Code file details
Three code files written in Python:

1. [train_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/train_NN_script.py): can be used to train any new networks with additional data
2. [predict_NN_script.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/predict_NN_script.py): Predict with existing neural networks
3. [general_functions.py](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/general_functions.py): function definitions

## Examples

Some examples showing accuracy of neural network prediction by superposing it with experimental data

1. Super-hydrophobic surface with seperation 4 mm and surface temperature -15 <sup>0</sup>C
![](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/Neural%20Networks/SHP_4mm_-15C%20Time%5Bmin%5D/superposed.png)

2. Super-hydrophilic surface with seperation 8 mm and surface temperature -10 <sup>0</sup>C
![](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/Neural%20Networks/SHL_8mm_10C/superposed.png)

3. Regular surface with seperation 2 mm and surface temperature -5 <sup>0</sup>C
![](https://github.com/shahaneshantanu/frost-thickness-neural-network/blob/master/Neural%20Networks/R_2mm_-5C%20%5Bmin%5D/superposed.png)

Similar other files can be found in the [folders](https://github.com/shahaneshantanu/frost-thickness-neural-network/tree/master/Neural%20Networks)

## Credits:

* Neural networks developed by Dr. Shantanu Shahane, Postdoctoral Research Associate, Mechanical Science and Engineering, University of Illinois at Urbana-Champaign. [Github Link](https://github.com/shahaneshantanu). Email: <shahaneshantanu@gmail.com>
* Experimental data from research group of [Prof. Nenad Miljkovic](http://etrl.mechanical.illinois.edu/). Email: <nmiljkov@illinois.edu>:

  * Kazi Fazle Rabbi, Graduate Research Assistant, MechSE, UIUC. Email: <kazif2@illinois.edu>
  * Kalyan Boyina, Graduate Research Assistant, MechSE, UIUC. Email: <boyina2@illinois.edu>
  * Anand Thamban, Visiting Scholar, MechSE, UIUC. Email: <anandthamban@gmail.com>
  * Wei Su, Visiting Scholar, MechSE, UIUC. Email: <weisu@illinois.edu>
  * Soumyadip Sett, Postdoctoral Research Associate, MechSE, UIUC. Email: <ssett3@illinois.edu>
