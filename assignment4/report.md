## Question 1

For SKLearn LinearRegression : 
- RMSE:  1.0002249324671484
- MAE:  0.7983290434730975
- MSE:  1.0004499155289115
- SKLearn Mean Time: 1.4644753217697144 
---------------------------
For linear regression using normal equations -
- RMSE:  1.0002249324671484
- MAE:  0.7983290434730974
- MSE:  1.0004499155289115
- Normal Equation Time: 0.182195782661438  
---------------------------
For linear regression using SVD - 

- RMSE -  1.0002249324671484
- MAE:  0.7983290434730975
- MSE:  1.0004499155289115
- SVD Mean Time: 1.868654990196228 
---------------------------

## Question 2

### Batch Gradient Descent

|Grad Type|Iteration Number             |Lr    |Penalty type                                 |Time          |RMSE       |MAE        |
|---------|-----------------------------|------|---------------------------------------------|--------------|-----------|-----------|
|manual   |50                           |0.005 |                                             |0.1068477631  |3.847907145|3.040924894|
|jax      |50                           |0.005 |                                             |3.279570818   |3.84792    |3.041896   |
|manual   |50                           |0.005 |l2                                           |0.1087119579  |3.848380961|3.044735703|
|jax      |50                           |0.005 |l2                                           |4.069816351   |3.848854   |3.046539   |
|manual   |5                            |0.005 |                                             |0.01097297668 |4.15384517 |3.38199367 |
|jax      |5                            |0.005 |                                             |0.3465218544  |4.161519   |3.3900337  |
|jax      |5                            |0.005 |l2                                           |0.4126756191  |4.17675    |3.405737   |
|manual   |5                            |0.005 |l2                                           |0.01137590408 |4.195160146|3.424294765|
|jax      |1                            |0.005 |l2                                           |0.08861589432 |11.346141  |10.871823  |
|manual   |1                            |0.005 |                                             |0.002527236938|11.35925478|10.88482305|
|manual   |1                            |0.005 |l2                                           |0.002524614334|11.37213089|10.89758476|
|jax      |1                            |0.005 |                                             |0.06935405731 |11.38932   |10.91462   |
|manual   |50                           |1.00E-05|                                             |0.109960556   |16.8997205 |16.26412785|
|jax      |50                           |1.00E-05|                                             |3.412736654   |16.899721  |16.264128  |
|manual   |50                           |1.00E-05|l2                                           |0.1084005833  |16.89979353|16.2641978 |
|jax      |50                           |1.00E-05|l2                                           |3.601226091   |16.899794  |16.264198  |
|manual   |5                            |1.00E-05|                                             |0.01099181175 |17.56108975|16.89689897|
|jax      |5                            |1.00E-05|                                             |0.3409428596  |17.56109   |16.8969    |
|jax      |5                            |1.00E-05|l2                                           |0.4182481766  |17.56109   |16.896902  |
|manual   |5                            |1.00E-05|l2                                           |0.01102733612 |17.56109029|16.89689949|
|manual   |1                            |1.00E-05|l2                                           |0.002566099167|17.62124591|16.95439213|
|manual   |1                            |1.00E-05|                                             |0.01126909256 |17.62124597|16.95439218|
|jax      |1                            |1.00E-05|                                             |0.8515610695  |17.621246  |16.954393  |
|jax      |1                            |1.00E-05|l2                                           |0.2187981606  |17.621248  |16.954393  |
|jax      |1                            |0.1   |                                             |0.06859016418 |32.897556  |31.386879  |
|jax      |1                            |0.1   |l2                                           |0.08511185646 |41.79998   |39.73423   |
|manual   |1                            |0.1   |                                             |0.002373695374|42.73844239|40.61307852|
|manual   |1                            |0.1   |l2                                           |0.002375364304|44.86462512|42.60362895|
|jax      |5                            |0.1   |l2                                           |0.4187202454  |776.9825   |725.5124   |
|manual   |5                            |0.1   |                                             |0.01104164124 |889.4623488|830.4107749|
|jax      |5                            |0.1   |                                             |0.3336157799  |1020.5567  |952.6685   |
|manual   |5                            |0.1   |l2                                           |0.01106929779 |1247.595821|1164.403572|
|manual   |50                           |0.1   |                                             |0.1049478054  |2.26E+18   |2.11E+18   |
|manual   |50                           |0.1   |l2                                           |0.1088733673  |1.65E+19   |1.53E+19   |
|jax      |50                           |0.1   |                                             |3.074924707   |inf        |2.98E+18   |
|jax      |50                           |0.1   |l2                                           |3.948280573   |inf        |5.04E+18   |

### SGD With Momemtum

|Iteration Number|Lr                           |Penalty type|Beta                                         |Time          |RMSE       |MAE        |
|----------------|-----------------------------|------------|---------------------------------------------|--------------|-----------|-----------|
|50              |1.00E-05                     |            |0.8                                          |2.722150803   |3.865948762|3.072027229|
|5               |1.00E-05                     |l2          |100                                          |2.251496315   |3.867575974|3.074571511|
|5               |0.1                          |            |100                                          |2.829063892   |3.976750622|3.190309327|
|1               |0.005                        |            |0.8                                          |3.02590847    |3.983070371|3.136690158|
|50              |0.005                        |            |100                                          |2.923275471   |4.091753625|3.211368433|
|50              |0.005                        |            |0.8                                          |3.62349391    |4.230902861|3.310163851|
|1               |0.005                        |l2          |100                                          |2.1452384     |4.276513235|3.505246169|
|5               |0.005                        |l2          |100                                          |2.52077961    |4.498743184|3.723124455|
|50              |0.1                          |            |0.8                                          |2.806585789   |4.870423239|4.120416779|
|50              |0.1                          |l2          |0.8                                          |3.13413167    |5.859663427|5.168984837|
|5               |0.005                        |            |0.8                                          |2.278110743   |8.273200776|6.773594797|
|5               |1.00E-05                     |l2          |0.8                                          |2.234700203   |9.798247197|8.09979766 |
|5               |0.1                          |l2          |100                                          |2.684328318   |11.43626927|9.557093357|
|5               |0.1                          |l2          |0.8                                          |2.821309566   |11.68098348|9.777664034|
|5               |0.1                          |            |0.8                                          |2.580688715   |11.72744023|9.819472682|
|1               |0.1                          |            |0.8                                          |2.096777678   |12.10899509|11.62499785|
|1               |0.005                        |l2          |0.8                                          |2.0965271     |12.787193  |10.7701554 |
|1               |1.00E-05                     |            |100                                          |3.926039934   |15.08893344|12.84023083|
|1               |1.00E-05                     |l2          |100                                          |3.667466402   |16.08000325|15.47795192|
|5               |0.005                        |l2          |0.8                                          |2.500407934   |16.25087317|15.64201886|
|5               |1.00E-05                     |            |100                                          |2.222960472   |18.85977174|16.31684633|
|50              |0.1                          |            |100                                          |3.424068451   |20.06470088|17.44889882|
|50              |1.00E-05                     |            |100                                          |2.724789143   |21.33456611|18.65240911|
|5               |0.005                        |            |100                                          |2.352838755   |22.17528129|21.28429485|
|50              |0.1                          |l2          |100                                          |2.76093173    |26.24725634|23.2957363 |
|1               |0.1                          |            |100                                          |2.138540506   |27.48090648|26.29349664|
|1               |0.005                        |            |100                                          |2.132934809   |28.1397035 |26.91384971|
|50              |0.005                        |l2          |0.8                                          |4.24572587    |28.55079022|25.4655249 |
|50              |0.005                        |l2          |100                                          |2.857210398   |34.22646944|30.79915548|
|50              |1.00E-05                     |l2          |100                                          |2.789741278   |37.43242084|35.6416928 |
|50              |1.00E-05                     |l2          |0.8                                          |3.31942749    |37.86966584|36.05160814|
|1               |1.00E-05                     |            |0.8                                          |3.988003969   |40.05129043|38.09616662|
|1               |0.1                          |l2          |100                                          |2.164969921   |43.31499658|41.15292862|
|1               |0.1                          |l2          |0.8                                          |2.231053829   |45.2980362 |43.00930291|
|1               |1.00E-05                     |l2          |0.8                                          |3.091001034   |55.1726775 |50.41038464|
|5               |1.00E-05                     |            |0.8                                          |2.163731337   |56.70341727|53.67660373|

## Question 3

Line Fit GIF:-

![line_fit](https://user-images.githubusercontent.com/62815174/228493635-aaf32ea4-7894-462b-9f41-e908617ad3b7.gif)

Contour Plot GIF:-

![contour_plot](https://user-images.githubusercontent.com/62815174/228493523-162a5ace-b699-4597-a24e-e7ff6e9c59f2.gif)

Surface Plot GIF:-

![surface_plot](https://user-images.githubusercontent.com/62815174/228493737-2a7fcdbe-cf91-4ee3-a2eb-2131d6c5c4ed.gif)

## Question 4

![thetanorm_degree](https://user-images.githubusercontent.com/62815174/228493962-185d750d-8cd3-45de-a010-a2d547a67d0d.png)

## Question 5

![thetanorm_degree](https://user-images.githubusercontent.com/62815174/228494037-f8ace31c-1b9d-48a7-b2b2-8b11fdd33e0a.png)

## Question 6

![Question6](https://user-images.githubusercontent.com/62815174/228538712-52ee27e8-4e8b-4afb-97a2-34e6e650bd05.png)

- RMSE:  2.888412497482042
- MAE:  2.263949518495599
- Scaled RMSE:  2.3887808528585697
- Scaled MAE:  1.8723361937864231

## Question 7

RMSE:- 3.5514716737310414

![Question7](https://user-images.githubusercontent.com/62815174/228494171-8ac63205-9b81-4dc3-947c-064b0a9a5def.png)

