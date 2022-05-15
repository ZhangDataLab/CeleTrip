# Visualization of Politicians' trip

## Intro

We develop a visualization tool to show the results of the proposed trip detection method. Due to the demand for Anonymity, here we provide the source code of the visualization tool, which is developed by Vuejs, and remove the API interface from the codes as well.

## Display

Since we anonymized the API interface, the process of interactive visualization can not work locally. The integral effect of this tool is shown in the *.gif* below.

### Features

1. Users can choose a politician and the period they are interested in.
2. Click the 'Query' button, and a map with the timeline will be rendered below, showing this politician's trips by time. 
3. Double click a target location, and users can view this politician's trip records to this place, while 'Wiki Label' indicates whether a record is included in Wikipedia.
4. Unfold the drop-down menu, and focal sentences from news related to this trip are given as well as their sources link. Users can also valid this trip by themselves.

![demo_of_viz_tool](demo/viz_tool.gif)

As shown in the picture, we choose *Donald Trump* and his trip records from 2019.07 to 2020.12. Then we view Trump's trips to West Palm Beach and check the news related to a trip in 2019.12.20.