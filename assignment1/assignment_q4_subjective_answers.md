Theoretical time complexity :- O(nlogn*d), for n data points and d features. Here we have taken n = d = 5, 20, 100. Theoretically the timings would be O(17.4), O(520.41), and O(20,000) respectively.

For, preditction the time complexity will be O(depth), so the theoretical timings would be O(2), O(7) and O(20).

The pratical timings for depths 2, 7 and 20 for the four different input cases are listed below:-

### Discrete Input Discrete Output

#### Fit Timings

|  | 5 | 20 | 100 |         
|--|--|--|--|
| 2 | 0.0003995180130004883 | 1.8218745231628417 | 7.012724661827088 |
| 7 | 0.0005247592926025391 | 3.0009424924850463 | 54.86989665031433 |
| 20 | 0.0003540515899658203 | 2.276933193206787 | 52.90173387527466 |

![DIDO_Fit](https://user-images.githubusercontent.com/62815174/213990974-472b6f5e-58d2-413a-bd6f-0123a3e9c8a1.png)

#### Preditction Timings

|  | 5 | 20 | 100 |         
|--|--|--|--|
| 2 | 0.00031812251315445323 | 0.008382461839402285 | 0.02408716859671269 |
| 7 | 0.0007144430932318932 | 0.009551184073620525 | 0.01621000535093641 |
| 20 | 0.0003645091793630074 | 0.0005438034717835922 | 0.042584806700167406 |

![DIDO_Predict](https://user-images.githubusercontent.com/62815174/213991014-71c205b5-2eb1-4496-98a6-89e38016c176.png)

### Discrete Input Real Output

#### Fit Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.031970763206481935 | 0.10993857383728027 | 0.4571547269821167 |
| 7 | 0.03243703842163086 | 0.49663984775543213 | 12.621706557273864 |
| 20 | 0.03816187381744385 | 0.5009801864624024 | 13.440599584579468 |

![DIRO_Fit](https://user-images.githubusercontent.com/62815174/213991072-6cd9aaae-5ea7-474a-be05-01f006c9f634.png)

#### Prediction Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.0004671748154308518 | 0.008311516457689528 | 0.01293462085471222 |
| 7 | 0.0004190722395775972 | 0.005635077763932613 | 0.10478125234996799 |
| 20 | 0.0005048199241888118 | 0.0018793058452189664 | 0.02534099184753767 |

![DIRO_Predict](https://user-images.githubusercontent.com/62815174/213991106-9c1dbda0-47f7-4420-81ef-b48937c3d3e5.png)

### Real Input Discrete Output

#### Fit Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.5938871145248413 | 15.24015154838562 | 450.6439693450928 |
| 7 | 0.558036994934082 | 23.029045128822325 | 1259.4093257427216 |
| 20 | 0.5234464406967163 | 19.44146022796631 | 1145.5983320236205 |

![RIDO_Fit](https://user-images.githubusercontent.com/62815174/213991136-6057ff47-93d9-4629-8d06-8ce1b8ab07b7.png)

#### Prediction Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.00014304137196518682 | 0.0003057990275846546 | 0.0009357530245202434 |
| 7 | 1.937785778197433e-05 | 0.0001287253261621685 |  0.0018539098055514462 |
| 20 | 1.5033402163665349e-05 | 8.838927582077e-05 |  0.001035221686313047 |

![RIDO_Predict](https://user-images.githubusercontent.com/62815174/213991159-eaa9c4dc-2ee9-417e-b187-cfcee872c953.png)

### Real Input Real Output

#### Fit Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.10094795227050782 | 1.7241599321365357 | 44.52255687713623 |
| 7 | 0.12138822078704833 | 3.8686156988143923 | 135.03472192287444 |
| 20 | 0.11461684703826905 | 3.5171552896499634 | 184.1080975294113 |

![RIRO_Fit](https://user-images.githubusercontent.com/62815174/213991185-d1322e19-8354-435c-a805-f9265ba44650.png)

#### Prediction Timings

|  | 5 | 20 | 100 | 
|--|--|--|--|
| 2 | 0.00027777447567645336 | 0.00040024414136924 | 0.0013592822931221761 |
| 7 | 5.0942664086302124e-05 | 0.0005614763212302351 | 0.0009986240598519414 |
| 20 | 0.00031380495425471484 | 8.368939064458897e-05 | 0.0013573435389717553 |

![RIRO_Predict](https://user-images.githubusercontent.com/62815174/213991206-47b591ef-fd35-43f4-a218-92db45274024.png)

### Observation and Inference
For tree fitting, we can see that the theoretical and experimental runtime ratios are of same order. For predict, the runtime ratios even theoretically were very minute and experimentally we observed the same, as there was not much difference observed.
