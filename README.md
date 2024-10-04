# JEB_2024
Python codes used to analyze fly heading data and create plots used in Pae, Liao, et al., 2024 

circstats.py is required to run all the data analysis code which can be found here:
https://github.com/jhamrick/python-snippets/blob/master/snippets/circstats.py

Each analysis code is organized by figures and subfigures. Each python code can be used to analyze the corresponding individual heading data available in Dryad(_).  

*Please note below:
- fig2_flight_data_analysis.py is for analyzing the heading of the groups where flies were flying during the varied stimulus period (fig2E,I, fig2F,J,   fig2M,Q and  fig2N,R) The first 15minutes of individual heading data from fig1B,C and fig1D is used to generate fig2F,J and fig2N,R, respectively.
-fig2_rest_data_analysis.py is for analyzing the heading of the groups where flies were given a piece of kimwipe during the varied stimulus period (fig2C,G, fig2D,H, fig2K,O, and fig2L,P)
- A short code is included at the end of fig2_flight_data_analysis.py and fig2_rest_data_analysis.py to generate csv files for GLM analysis in figure 2. A separate code (combine_csv_for_glm.py) is used to combine all the generated csv files into one csv files for a GLM analysis
- Figure 10 analysis is included in figure5_data_analysis.py (Fig10B) and figure9_data_analysis.py(Fig10C) as Figure 10 is an additional analysis for flies' headings from Figure 5 and Figure 9.
