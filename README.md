# Titanic survival project

<p align="center">
<img src="images/Titanic.jpg" width="500">
</p>


RMS Titanic was a British passenger liner, operated by the White Star Line, which sank in the North Atlantic Ocean on 15 April 1912 after striking an iceberg during her maiden voyage from Southampton, UK, to New York City. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew which made the sinking possibly one of the deadliest for a single ship up to that time. It remains to this day the deadliest peacetime sinking of a superliner or cruise ship.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others. This project aims to develop a machine learning classifiers to predict which passengers survived the Titanic shipwreck given features like age, gender, socio-economic class, etc.

## About the data

The data was downloaded from <a href="https://www.kaggle.com/c/titanic">Kaggle</a> and only the trainning set is in use for this project. The trainning set containg the target variable and can be used to evaluate our model. The test set, without the target, is used for predictions when participating in the Kaggle competition.

<p align="center">
<img src="images/data_table.png" width="500">
</p>

**Variable Notes**

- `pclass`: A proxy for socio-economic status (SES)<br> 
 1st: Upper<br> 
 2nd: Middle<br>
 3rd: Lower<br>

- `age`: Age is fractional if less than 1. If the age is estimated, is it in the form of xx.5

- `sibsp`: The dataset defines family relations in this way:<br>
Sibling: brother, sister, stepbrother, stepsister<br>
Spouse: husband, wife (mistresses and fiancés were ignored)<br>

- `parch`: The dataset defines family relations in this way:<br>
Parent: mother, father<br>
Child: daughter, son, stepdaughter, stepson<br>
Some children travelled only with a nanny, therefore parch=0 for them.<br>
