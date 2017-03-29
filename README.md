# RPI_Datathon17

Team members:
Han Wang,
Li Dong,
Michael Agiorgousis.


In this Datathon, Area Healtsources Files (AHRF) from Health Resources and Services Administration (HRSA) for all counties in US was used and the objective is to predict hospital utilization. Detailed instructions of this Datathon can be seen in RPI_Datathon_2017_Logistics.pdf and RPI__Datathon_2017_Instructions.pdf.

We manually engineered about 500 features from about 6000 features and trained our model on about 2000 data points. We explored various models such as Lasso, SVM with linear and RBF kernels and gradient boosting / random forest. Due to a hard limit on training time - 10 min, we did not adopt more advanced nonlinear models. In the end, random forest was found to be robust and provide best prediction. eXtreme Gradient Boosting (XGB) was used to perform modeling. Our final deliverables include model.py and Team2_Datathon_17.pdf.
