# MAHANGA
MĀHANGA: Model Ānalysis: Handling biNary Grid VisuAlisation - designed for use with Aotearoa STARS

Māhanga is the Māori words for twins, which is an apt description for binary stars since they are (typically) born at the same time, from the same cloud of gas.

This analysis framework was designed to use information from the output files of an Aotearoa STARS simulation run, enabling a more streamlined experience when analysing a large grid of simulations. The framework can identify the simulation's final state, extract all relevant information from the out and plot files, and compile it into data frames. It also has the capability to clean and combine the stage one and stage two simulations, so that all the data for each star are contained in a single group. 

Once the appropriate data have been extracted, further information can be determined; for example, the effective Roche lobe of each star can be determined. It can also find the end of the main sequence by identifying when the helium core concentration reaches a certain level, and similarly, find the end of helium burning. The key feature of the framework is its creation of graphs for all simulations, which contain important information, such as H-R diagrams, mass evolution, radius evolution, and more. It can compile all the graphs and information for each system into a single file. 

In addition to the individual figures for each system, the framework could gather broad information from each simulation and use it to create a grid showing patterns across the full range of simulations. 

PLEASE NOTE I am not a data/computer scientist, and my coding skills are questionable at best. As it currently is, MĀHANGA requires a certain file structure, and the functions are very messy. There are many improvements to be made, but this is a start.
