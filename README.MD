# Discrete non-spatial navigation 

Data recorded from Vinny performing the 'Discrete Maze' [task](Task.pdf). A 9x12 maze where each state is an unrelated image connected to its 4 direct neighbours. On each trial Vinny is shown a target location, and must choose the correct states to lead him to the target location. He does this with high precision. 

Data consists of individual neurons, 162 from dorsal ACC (dACC), 143 from OFC, and 92 from Area 32. The data has been processed and is ready to use. Rasters have been smoothed and normalised, and consist of one row per trial, and one column per 10 ms of time. They are +/-1000ms of either the time at which the options were displayed on the screen ('optionsOn'), or the time at which Vinny chose one of the options ('optionMade'). 

There are various trial parameters, each in its own array:
   - distChange - For this step, what is the new distance to target compared to the previous step (+1 = moved towards target)
   - currAngle - Angle between current location and target
   - hd - What direction did they move (north south east west)
   - numsteps - In this trial, how many steps have they taken?
   - perfTrials - Was this trial perfect? I.e. all steps were towards the target
   - startAngle - Starting angle to the target
   - currDist - Current distance to the target
   - from_x - X-coordinate of state they just moved from
   - from_y - Y-coordinate of ....
   - to_x - X-coordinate of state they chosen on this trial
   - to_y - Y-coordinate of ...

Simply run [rate_maps.py](rate_maps.py) and it will generate various rate map plots for each neuron like the one below. The question is whether or not any of these are actual grid cells. 

![image](plots/plot_optionMade_Area32_17.png)